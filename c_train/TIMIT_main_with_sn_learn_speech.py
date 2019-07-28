import sys
import os
curPath = os.path.abspath(os.path.dirname(__file__))
rootPath = os.path.split(curPath)[0]
sys.path.append(rootPath)
import torch
import soundfile as sf
import librosa
import numpy as np
from a_net.module_with_SN_8k_2x import GeneratorNet, DiscriminatorNet
from b_load_data.SpeechDataLoad import SpeechDataLoader, SpeechDataset, FeatureCreator
import torch.optim as optim
from utils.model_handle import save_model, resume_model, save_discriminator_model
from utils.loss_set import LossHelper
from utils.pesq import pesq
import progressbar
from config import *
from utils.util import get_alpha, normalization, re_normalization, discriminator_regularizer
from utils.stft_istft import STFT
from tensorboardX import SummaryWriter
from utils.util import normalization, re_normalization, vec2frame


def validation_pesq(net, feature_creator):
    net.eval()
    net.cuda(CUDA_ID[0])
    files = os.listdir(VALIDATION_DATA_PATH)
    files.sort()
    stft = STFT(filter_length=FILTER_LENGTH, hop_length=HOP_LENGTH)
    bar = progressbar.ProgressBar(0, len(files))
    bar.start()
    base_pesq = 0
    res_pesq = 0
    promote_pesq = 0
    sum_seg_snr = 0
    sum_full_lsd = 0
    sum_half_lsd = 0
    noise_list = np.load('noise_4_TIMIT_new.npy', allow_pickle=True)
    loss_helper = LossHelper()
    if NEED_NORM:
        if IS_LOG:
            train_mean = feature_creator.log_train_mean
            train_var = feature_creator.log_train_var
        else:
            train_mean = feature_creator.train_mean
            train_var = feature_creator.train_var
    tag = 0
    for i in range(len(files)):
        if not files[i].endswith('_noise.wav') and files[i].endswith('.wav'):
            bar.update(i)
            # input
            speech, sr = sf.read(VALIDATION_DATA_PATH + files[i])
            # 对齐
            nframe = (len(speech) - FILTER_LENGTH) // HOP_LENGTH + 1
            size = nframe % 32
            nframe = nframe - size
            sample_num = FILTER_LENGTH + (nframe - 1) * HOP_LENGTH
            speech = speech[:sample_num]
            # stft
            speech = torch.Tensor(speech).unsqueeze(0)
            speech_spec = stft.transform(speech)
            speech_real = speech_spec[:, :, :, 0]
            speech_imag = speech_spec[:, :, :, 1]
            speech_mag = torch.sqrt(speech_real ** 2 + speech_imag ** 2).cuda(CUDA_ID[0])
            speech_mag = speech_mag.permute(0, 2, 1).cuda(CUDA_ID[0])
            # label，为了计算pesq
            clean, sr = sf.read(VALIDATION_DATA_LABEL_PATH + files[i])
            clean = clean[:sample_num]
            noise = noise_list[i][:sample_num]
            # add noise for calculate real pesq
            clean += noise
            # for calculate seg_snr
            clean_frame = vec2frame(clean, 32, 8)
            sf.write('clean_cat.wav', clean, sr)
            # 低频部分的语音，最后cat在一起用
            base, sr = sf.read(VALIDATION_DATA_PATH + files[i])
            base = base[:sample_num]
            # tmp 为半波整流之后获取相位用
            tmp = base
            tmp_low = base.copy()
            base += noise
            sf.write('base_cat.wav', base, sr)
            base_mag = stft.spec_transform(stft.transform(torch.Tensor(base[np.newaxis, :]))).squeeze()
            base_mag = base_mag.permute(1, 0)
            p1 = pesq('clean_cat.wav', 'base_cat.wav', is_current_file=True)
            clean = torch.Tensor(clean).unsqueeze(0)
            clean_spec = stft.transform(clean)
            clean_real = clean_spec[:, :, :, 0]
            clean_imag = clean_spec[:, :, :, 1]
            clean_mag = torch.sqrt(clean_real ** 2 + clean_imag ** 2).cuda(CUDA_ID[0]).squeeze()
            # 切片
            speech_mag = speech_mag.squeeze()
            base_list = []
            input_list = []
            clean_list = []
            for k in range(int(speech_mag.size()[1] / 32)):
                k *= 32
                item = speech_mag[:, k:k + 32].cpu().detach().numpy()
                base_item = base_mag[:, k:k + 32].cpu().detach().numpy()
                clean_item = clean_mag[k:k+32, :].cpu().detach().numpy()
                base_list.append(base_item)
                input_list.append(item)
                clean_list.append(clean_item)
            base_list = torch.Tensor(np.array(base_list)).cuda(CUDA_ID[0])
            input_list = torch.Tensor(np.array(input_list)).cuda(CUDA_ID[0])
            clean_list = torch.Tensor(np.array(clean_list)).cuda(CUDA_ID[0])
            if IS_LOG:
                input_list = torch.log(input_list + torch.Tensor(np.array(EPSILON)).cuda(CUDA_ID[0]))
            # normalized
            if NEED_NORM:
                input_list = ((input_list.permute(0, 2, 1) - train_mean) / (train_var + torch.Tensor(np.array(EPSILON)).cuda(CUDA_ID[0]))).permute(0, 2, 1)
            # 送入网络
            est_speech = net(input_list[:, :65, :])
            if IS_LOG:
                est_speech = torch.exp(est_speech)
            # 恢复语音
            est_mag = torch.cat((base_list.permute(0, 2, 1)[:, :, :58], est_speech), 2)
            # 合并
            half_lsd = loss_helper.mertics_LSD(est_mag, clean_list, is_full=False)
            full_lsd = loss_helper.mertics_LSD(est_mag, clean_list, is_full=True)
            est_mag = est_mag.reshape(-1, est_mag.shape[2]).unsqueeze(0)
            "end"

            sum_full_lsd += full_lsd.item()
            sum_half_lsd += half_lsd.item()
            # 获取半波整流后的相位
            tmp[tmp < 0] = 0
            tmp = torch.Tensor(tmp).unsqueeze(0)

            tmp_low_spec = stft.transform(torch.Tensor(tmp_low[np.newaxis, :]))
            tmp_low_real = tmp_low_spec[:, :, :, 0]
            tmp_low_imag = tmp_low_spec[:, :, :, 1]
            tmp_low_mag = torch.sqrt(tmp_low_real ** 2 + tmp_low_imag ** 2).squeeze()
            "end"

            # 获取相位信息
            tmp_spec = stft.transform(tmp)
            tmp_real = tmp_spec[:, :, :, 0]
            tmp_imag = tmp_spec[:, :, :, 1]
            tmp_mag = torch.sqrt(tmp_real ** 2 + tmp_imag ** 2)
            # 低频使用原始相位，高频使用半波整流之后的语音的相位
            tmp_mag = torch.cat((tmp_low_mag[:, :65], tmp_mag.squeeze()[:, 65:]), 1)
            tmp_real = torch.cat((tmp_low_real.squeeze()[:, :65], tmp_real.squeeze()[:, 65:]), 1)
            tmp_imag = torch.cat((tmp_low_imag.squeeze()[:, :65], tmp_imag.squeeze()[:, 65:]), 1)
            # TODO test
            # test_real = test.detach().cpu() * tmp_real / tmp_mag
            # test_imag = test.detach().cpu() * tmp_imag / tmp_mag
            # test_res = torch.stack([test_real, test_imag], 3)
            # test_res = stft.inverse(test_res)
            # sf.write('test.wav', test_res.numpy().squeeze(), sr)
            # 恢复语音
            res_real = est_mag.detach().cpu() * tmp_real / tmp_mag
            res_imag = est_mag.detach().cpu() * tmp_imag / tmp_mag
            res_spec = torch.stack([res_real, res_imag], 3)
            res = stft.inverse(res_spec)
            # sf.write(files[i][:-4] + '_res.wav', res.squeeze().detach().numpy(), sr)
            # 加噪
            res += torch.Tensor(noise[: res.shape[1]])
            sf.write('res_cat.wav', res.numpy().squeeze(), sr)
            p2 = pesq('clean_cat.wav', 'res_cat.wav', is_current_file=True)
            res_frame = vec2frame(res, 32, 8)
            loss_helper = LossHelper()
            seg_snr = loss_helper.seg_SNR_(torch.Tensor(res_frame), torch.Tensor(clean_frame))

            sum_seg_snr += seg_snr.item()
            base_pesq += p1[0]
            res_pesq += p2[0]
            promote = p2[0] - p1[0]
            promote_pesq += promote
    bar.finish()
    return base_pesq / VALIDATION_DATA_NUM, res_pesq / VALIDATION_DATA_NUM, promote_pesq / VALIDATION_DATA_NUM, sum_seg_snr / VALIDATION_DATA_NUM, sum_half_lsd / VALIDATION_DATA_NUM, sum_full_lsd / VALIDATION_DATA_NUM


def train(g_net, d_net, g_opt, d_opt, epoch, data_loader, loss_helper):
    global global_step
    path_dir = 'TIMIT_train_learn_speech_with_sn/'
    # create log and module store
    if not os.path.exists(LOG_STORE + path_dir):
        os.mkdir(LOG_STORE + path_dir)
    if not os.path.exists(MODEL_STORE + path_dir):
        os.mkdir(MODEL_STORE + path_dir)
    log_store = LOG_STORE + path_dir
    module_store = MODEL_STORE + path_dir
    # LAMBDA_FOR_REC_LOSS
    lambda_for_rec_loss = 10
    writer = SummaryWriter(log_store)
    feature_creator = FeatureCreator()
    bar = progressbar.ProgressBar(0, train_cdnn_data_set.__len__() // TRAIN_BATCH_SIZE)
    g_sum_loss = 0
    sum_real_loss = 0
    sum_fake_loss = 0
    sum_reconstruction_loss = 0
    sum_adversarial_loss = 0
    sum_disc_reg = 0
    for i in range(epoch):
        # progressbar
        bar.start()
        for batch_idx, batch_info in enumerate(data_loader.get_data_loader()):
            bar.update(batch_idx)
            # g_input、g_label (B, T, F) = (64, 32, 129)
            g_input, g_label = feature_creator(batch_info)
            # noinspection PyUnresolvedReferences
            g_label = torch.autograd.Variable(g_label, requires_grad=True)
            # fake_input = torch.autograd.Variable(fake_input, requires_grad=True)

            "1. train d_net"
            # 1.1 real
            d_opt.zero_grad()
            # d_net输入的是归一化后的g_input和g_label cat在一起的，即32 * (K + N)
            speech_real_input = torch.cat((g_input[:, :, :65], g_label[:, :, 58:]), 2)
            speech_real_input = speech_real_input.permute(0, 2, 1)
            speech_real_input = speech_real_input.unsqueeze(3)
            # input: B, F, T, 1
            # output: B, 1
            logits_real = d_net(speech_real_input)
            real_loss = loss_helper.discriminator_loss_with_sigmoid(logits_real, is_fake=False)
            real_loss.backward(retain_graph=True)
            sum_real_loss += real_loss.item()


            # 1.2 fake
            # g_net输入的是经过半波整流之后的低频语音
            # B, F, T
            est_speech = g_net(g_input[:, :, :65].permute(0, 2, 1))
            # fake_input = torch.cat((speech_low_pass_mag[:, :, 0:35], est_speech), 2)
            # fake_input = torch.autograd.Variable(fake_input, requires_grad=True)
            # input to d_net, fake_input shape is (B,F,T,1), especially F is K + N = 65 + 71 = 136
            fake_input = torch.cat((g_input[:, :, :65], est_speech), 2).permute(0, 2, 1).unsqueeze(3)
            logits_fake = d_net(fake_input)
            fake_loss = loss_helper.discriminator_loss_with_sigmoid(logits_fake, is_fake=True)
            # disc_reg = discriminator_regularizer(logits_real, real_label, logits_fake, fake_input, d_net, d_opt)
            # sum_disc_reg += disc_reg.item()
            # disc_loss = real_loss + fake_loss + (GAMMA / 2.0) * disc_reg
            # disc_loss.backward()

            fake_loss.backward()
            sum_fake_loss += fake_loss.item()
            d_opt.step()
            "end"

            "2. train g_net"
            g_opt.zero_grad()
            est_speech = g_net(g_input[:, :, :65].permute(0, 2, 1))
            # fake_input = torch.cat((speech_low_pass_mag[:, :, 0:35], est_speech), 2)
            fake_input = torch.cat((g_input[:, :, :65], est_speech), 2).permute(0, 2, 1).unsqueeze(3)
            logits_fake = d_net(fake_input)
            adversarial_loss = loss_helper.generator_loss_with_sigmoid(logits_fake)
            reconstruction_loss = loss_helper.MSE_loss(est_speech, g_label[:, :, 58:])
            g_loss = adversarial_loss + lambda_for_rec_loss * reconstruction_loss
            g_loss.backward()
            g_opt.step()
            "end"
            sum_reconstruction_loss += (lambda_for_rec_loss * reconstruction_loss).item()
            sum_adversarial_loss += adversarial_loss.item()
            g_sum_loss += g_loss.item()
            if global_step % PRINT_TIME == 0 and global_step != 0:
                writer.add_scalar('Generator/generator_loss', g_sum_loss / PRINT_TIME, global_step)
                writer.add_scalar('Generator/reconstruction_loss', sum_reconstruction_loss / PRINT_TIME, global_step)
                writer.add_scalar('Generator/adverarial_loss', sum_adversarial_loss / PRINT_TIME, global_step)
                writer.add_scalar('Loss/real_loss', sum_real_loss / PRINT_TIME, global_step)
                writer.add_scalar('Loss/fake_loss', sum_fake_loss / PRINT_TIME, global_step)
                writer.add_scalar('Loss/disc_reg', sum_disc_reg / PRINT_TIME, global_step)
                sum_reconstruction_loss = 0
                sum_adversarial_loss = 0
                g_sum_loss = 0
                sum_fake_loss = 0
                sum_real_loss = 0
                sum_disc_reg = 0
            if global_step % SAVE_TIME == 0:
                save_model(global_step, generator, g_opt, g_loss.item(), module_store + 'g_model_' + str(global_step) + '.pkl')
                save_discriminator_model(global_step, discriminator, discriminator_opt, real_loss.item(), fake_loss.item(), module_store + 'd_model_' + str(global_step) + '.pkl')
                base_pesq, res_pesq, promote_pesq, seg_snr, half_lsd, full_lsd = validation_pesq(g_net, feature_creator)
                writer.add_scalar('SegSNR/seg_snr', seg_snr, global_step)
                writer.add_scalar('LSD/half_lsd', half_lsd, global_step)
                writer.add_scalar('LSD/full_lsd', full_lsd, global_step)
                writer.add_scalar('PESQ/base_pesq', base_pesq, global_step)
                writer.add_scalar('PESQ/res_pesq', res_pesq, global_step)
                writer.add_scalar('PESQ/promote_pesq', promote_pesq, global_step)
                g_net.train()
            global_step += 1
        bar.finish()


pre_train_global_step = 0
global_step = 0
if __name__ == "__main__":
    torch.manual_seed(0)
    torch.backends.cudnn.deterministic = True
    torch.backends.cudnn.benchmark = False
    # 初始化训练集
    train_cdnn_data_set = SpeechDataset(TRAIN_DATA_PATH, TRAIN_LABEL_PATH)
    train_data_loader = SpeechDataLoader(data_set=train_cdnn_data_set,
                                         batch_size=TRAIN_BATCH_SIZE,
                                         is_shuffle=True)
    pre_train_g_tag = False
    pre_train_d_tag = False
    train_tag = True

    # "pre train generator"
    # if pre_train_g_tag:
    #     net = GeneratorNet()
    #     # res = resume_model(net, MODEL_STORE + 'IEEE_pre_train_g_learn_speech_with_noise/model_666000.pkl')
    #     net = net.cuda(CUDA_ID[0])
    #     train_loss_helper = LossHelper()
    #     train_optimizer = optim.Adam(net.parameters(), lr=1e-5)
    #     # base_pesq, res_pesq, promote_pesq, seg_snr, half_lsd, full_lsd = validation_pesq(net)
    #     # print('base pesq : ' + str(base_pesq))
    #     # print('res pesq : ' + str(res_pesq))
    #     # print('promote pesq : ' + str(promote_pesq))
    #     # print('seg snr : ' + str(seg_snr))
    #     # print('full lsd : ' + str(full_lsd))
    #     # print('half lsd : ' + str(half_lsd))
    #     pre_train_g(net, EPOCH, train_data_loader, train_loss_helper, train_optimizer)
    # "pre train discrimintor"
    # if pre_train_d_tag:
    #     generator = GeneratorNet()
    #     res = resume_model(generator, MODEL_STORE + 'g_model_pre_train.pkl')
    #     generator = generator.cuda(CUDA_ID[0])
    #     generator.eval()
    #     discriminator = DiscriminatorNet()
    #     discriminator = discriminator.cuda(CUDA_ID[0])
    #     train_loss_helper = LossHelper()
    #     train_optimizer = optim.Adam(discriminator.parameters(), lr=LR)
    #     # wrong_data_num, right_data_num = validation_d_model(generator, discriminator)
    #     pre_train_d(discriminator, generator, EPOCH, train_data_loader, train_loss_helper, train_optimizer)

    "train"
    if train_tag:
        loss_helper = LossHelper()
        # 生成器
        generator = GeneratorNet()
        # res = resume_model(generator, MODEL_STORE + 'IEEE_pre_train_g_learn_speech_with_noise_sigmoid_10/g_model_pre_train.pkl')
        # generator.train()
        generator = generator.train().cuda(CUDA_ID[0])
        # TODO LR
        generator_opt = optim.Adam(generator.parameters(), lr=LR)

        # 判别器
        discriminator = DiscriminatorNet()
        # res = resume_model(discriminator, MODEL_STORE + 'd_model_pre_train.pkl')
        # discriminator.train()
        discriminator = discriminator.cuda(CUDA_ID[0])
        discriminator_opt = optim.Adam(discriminator.parameters(), lr=LR)
        train(generator, discriminator, generator_opt, discriminator_opt, EPOCH, train_data_loader, loss_helper)
