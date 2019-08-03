from config import *
import progressbar
from utils.stft_istft import STFT
import soundfile as sf
import numpy as np
from utils.model_handle import resume_model, resume_discriminator_model
from b_load_data.SpeechDataLoad import FeatureCreator
import torch
from utils.util import normalization, re_normalization, vec2frame
from utils.pesq import pesq
from scipy.io import loadmat, savemat
from utils.loss_set import LossHelper
from pystoi.stoi import stoi
from a_net.module_with_SN_16k_2x import GeneratorNet, DiscriminatorNet
import shutil
from tensorboardX import SummaryWriter


def validation_pesq(net, output_path):
    net.eval()
    net.cuda(CUDA_ID[0])
    files = os.listdir(VALIDATION_DATA_PATH)
    files.sort()
    stft = STFT(filter_length=FILTER_LENGTH, hop_length=HOP_LENGTH)
    bar = progressbar.ProgressBar(0, 100)
    bar.start()
    base_pesq = 0
    res_pesq = 0
    promote_pesq = 0
    sum_seg_snr = 0
    sum_full_lsd = 0
    sum_half_lsd = 0
    feature_creator = FeatureCreator()
    noise_list = np.load('noise_4_TIMIT_new.npy', allow_pickle=True)
    loss_helper = LossHelper()
    if NEED_NORM:
        train_mean = feature_creator.train_mean
        train_var = feature_creator.train_var
        train_label_mean = feature_creator.train_label_mean
        train_label_var = feature_creator.train_label_var
    for i in range(100):
        bar.update(i)
        # input
        speech, sr = sf.read(VALIDATION_DATA_PATH + files[i])
        # 对齐
        nframe = (len(speech) - FILTER_LENGTH) // HOP_LENGTH + 1
        size = nframe % 32
        nframe = nframe - size
        sample_num = FILTER_LENGTH + (nframe - 1) * HOP_LENGTH
        speech = speech[:sample_num]
        sf.write(output_path + files[i], speech, sr)
        # tmp 为半波整流之后获取相位用
        base = speech.copy()
        tmp = speech.copy()
        tmp_low = speech.copy()
        sf.write('base_cat.wav', base, sr)
        base_mag = stft.spec_transform(stft.transform(torch.Tensor(base[np.newaxis, :]))).squeeze()
        base_mag = base_mag.permute(1, 0)
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
        # noise = noise_list[i][:sample_num]
        # add noise for calculate real pesq
        # clean += noise
        # for calculate seg_snr
        clean_frame = vec2frame(clean, 32, 8)
        sf.write(output_path + files[i][:-4] + '_clean.wav', clean, sr)
        sf.write('clean_cat.wav', clean, sr)
        # 低频部分的语音，最后cat在一起用
        # base, sr = sf.read(VALIDATION_DATA_PATH + files[i])
        # sf.write(output_path + files[i][:-4] + '_base.wav', base, sr)
        # base = base[:sample_num]
        # base += noise

        p1 = pesq('clean_cat.wav', 'base_cat.wav', is_current_file=True)
        clean = torch.Tensor(clean).unsqueeze(0)
        clean_spec = stft.transform(clean)
        clean_real = clean_spec[:, :, :, 0]
        clean_imag = clean_spec[:, :, :, 1]
        clean_mag = torch.sqrt(clean_real ** 2 + clean_imag ** 2).cuda(CUDA_ID[0]).squeeze()
        # 切片
        speech_mag = speech_mag.squeeze()
        # input_list = speech_mag.reshape(-1, speech_mag.size()[1] // 32, 32)
        # base_list = base_mag.reshape(-1, speech_mag.size()[1] // 32, 32)
        # clean_list = clean_mag.reshape(-1, speech_mag.size()[1] // 32, 32)
        base_list = []
        clean_list = []
        input_list = []
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
        est_speech = net(input_list[:, :129, :])
        if NEED_NORM:
            est_speech = est_speech * train_label_var[116:] + train_label_mean[116:]
        if IS_LOG:
            est_speech = torch.exp(est_speech)
        est_mag = torch.cat((base_list.permute(0, 2, 1)[:, :, :116], est_speech), 2)
        # savemat('tmp.mat', {'est': est_speech.reshape(-1, 71).detach().cpu().numpy(), 'clean_mag': clean_mag[:, 116:].log().detach().cpu().numpy()})
        # savemat('tmp.mat', {'est_mag': est_mag.reshape(-1, 129).detach().cpu().numpy(), 'clean_mag': clean_mag.detach().cpu().numpy()})
        # est_mag = torch.cat((base_list.permute(0, 2, 1)[:, :, :116], clean_list[:, :, 116:]), 2)
        half_lsd = loss_helper.mertics_LSD(est_mag, clean_list, is_full=False)
        full_lsd = loss_helper.mertics_LSD(est_mag, clean_list, is_full=True)
        # 合并
        est_mag = est_mag.reshape(-1, est_mag.shape[2]).unsqueeze(0)
        "end"

        sum_full_lsd += full_lsd.item()
        sum_half_lsd += half_lsd.item()
        # 获取低频相位信息
        tmp_low_spec = stft.transform(torch.Tensor(tmp_low[np.newaxis, :]))
        tmp_low_real = tmp_low_spec[:, :, :, 0]
        tmp_low_imag = tmp_low_spec[:, :, :, 1]
        tmp_low_mag = torch.sqrt(tmp_low_real ** 2 + tmp_low_imag ** 2).squeeze()
        "end"

        # 获取高频相位信息
        tmp[tmp < 0] = 0
        tmp = torch.Tensor(tmp).unsqueeze(0)
        tmp_spec = stft.transform(tmp)
        tmp_real = tmp_spec[:, :, :, 0]
        tmp_imag = tmp_spec[:, :, :, 1]
        tmp_mag = torch.sqrt(tmp_real ** 2 + tmp_imag ** 2)
        # 低频使用原始相位，高频使用半波整流之后的语音的相位
        tmp_mag = torch.cat((tmp_low_mag[:, :129], tmp_mag.squeeze()[:, 129:]), 1)
        tmp_real = torch.cat((tmp_low_real.squeeze()[:, :129], tmp_real.squeeze()[:, 129:]), 1)
        tmp_imag = torch.cat((tmp_low_imag.squeeze()[:, :129], tmp_imag.squeeze()[:, 129:]), 1)
        # 纯净语音的相位
        # tmp_mag = clean_mag.detach().cpu()
        # tmp_real = clean_real.detach().cpu()
        # tmp_imag = clean_imag.detach().cpu()
        # TODO e_test
        # test_real = e_test.detach().cpu() * tmp_real / tmp_mag
        # test_imag = e_test.detach().cpu() * tmp_imag / tmp_mag
        # test_res = torch.stack([test_real, test_imag], 3)
        # test_res = stft.inverse(test_res)
        # sf.write('e_test.wav', test_res.numpy().squeeze(), sr)
        # 恢复语音
        res_real = est_mag.detach().cpu() * tmp_real / tmp_mag
        res_imag = est_mag.detach().cpu() * tmp_imag / tmp_mag
        res_spec = torch.stack([res_real, res_imag], 3)
        res = stft.inverse(res_spec)
        sf.write(output_path + files[i][:-4] + '_res.wav', res.squeeze().detach().numpy(), sr)
        # 加噪
        # res += torch.Tensor(noise[: res.shape[1]])
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


def validation_stoi(net):
    net.eval()
    net.cuda(CUDA_ID[0])
    files = os.listdir('/home/yangyang/userspace/data/TIMIT_low_pass/8k_new_2x_abs/e_test/')
    files.sort()
    sum_loss = 0
    stft = STFT(filter_length=FILTER_LENGTH, hop_length=HOP_LENGTH)
    bar = progressbar.ProgressBar(0, len(files))
    # bar = progressbar.ProgressBar(0, 100)
    bar.start()
    base_stoi = 0
    res_stoi = 0
    promote_stoi = 0
    time = 0
    noise_list = np.load('noise_4_TIMIT.npy', allow_pickle=True)
    p_sum = 0
    for i in range(len(files)):
        if files[i].endswith('_abs.wav'):
            bar.update(i)
            # 经过滤波器的语音
            # input
            speech, sr = sf.read('/home/yangyang/userspace/data/TIMIT_low_pass/8k_new_2x_abs/e_test/' + files[i])
            # 对齐∂
            nframe = (len(speech) - FILTER_LENGTH) // HOP_LENGTH + 1
            size = nframe % 32
            nframe = nframe - size
            sample_num = FILTER_LENGTH + (nframe - 1) * HOP_LENGTH
            speech = speech[:sample_num]
            # speech[speech < 0] = 0
            # speech = np.sqrt(speech)
            # input
            # sf.write('/home/yangyang/userspace/data/TIMIT_low_pass/8k/generator_data/' + files[i], tmp, sr)
            # speech, sr = sf.read('/home/yangyang/PycharmProjects/low_frequency_predict_high_frequency/train/REC40_R.wav')
            # label，为了计算loss
            clean, sr = sf.read('/home/yangyang/userspace/data/TIMIT_low_pass/8k_new_2x_abs/test_label/' + files[i][:-8] + '.wav')
            clean = clean[:sample_num]
            # s1 = stoi(clean, speech, sr)
            # sf.write('/home/yangyang/userspace/data/TIMIT_low_pass/8k/generator_data/' + files[i][:-4] + '_clean.wav',clean, sr)
            # 加噪
            noise = noise_list[time]
            time += 1
            clean += noise
            sf.write('clean.wav', clean, sr)
            base, sr = sf.read('/home/yangyang/userspace/data/TIMIT_low_pass/8k_new_2x_abs/e_test/' + files[i][:-8] + '.wav')
            base = base[:sample_num]
            tmp = base
            base += noise
            base_mag = stft.spec_transform(stft.transform(torch.Tensor(base[np.newaxis, :]))).squeeze()
            base_mag = base_mag.permute(1, 0)
            s1 = stoi(clean, base, sr)
            # sf.write('base.wav', base, sr)
            # p1 = pesq('clean.wav', 'base.wav', is_current_file=True)
            clean = torch.Tensor(clean).unsqueeze(0)
            clean_spec = stft.transform(clean)
            clean_real = clean_spec[:, :, :, 0]
            clean_imag = clean_spec[:, :, :, 1]
            clean_mag = torch.sqrt(clean_real ** 2 + clean_imag ** 2).cuda(CUDA_ID[0])
            # stft
            speech = torch.Tensor(speech).unsqueeze(0)
            speech_spec = stft.transform(speech)
            speech_real = speech_spec[:, :, :, 0]
            speech_imag = speech_spec[:, :, :, 1]
            speech_mag = torch.sqrt(speech_real ** 2 + speech_imag ** 2).cuda(CUDA_ID[0])
            # if IS_LOG:
            #     speech_mag = torch.log(speech_mag + EPSILON)
            speech_mag = speech_mag.permute(0, 2, 1).cuda(CUDA_ID[0])
            "方法一 切片："
            "start"
            # 切片
            speech_mag = speech_mag.squeeze()
            # speech_mag = torch.cat((base_mag[0:18, :].cuda(CUDA_ID[0]), speech_mag[18:81, :]), 0)
            base_list = []
            input_list = []
            for k in range(int(speech_mag.size()[1] / 32)):
                k *= 32
                item = speech_mag[:, k:k + 32].cpu().detach().numpy()
                base_item = base_mag[:, k:k + 32].cpu().detach().numpy()
                base_list.append(base_item)
                input_list.append(item)
            base_list = torch.Tensor(np.array(base_list)).cuda(CUDA_ID[0])
            input_list = torch.Tensor(np.array(input_list)).cuda(CUDA_ID[0])
            "end"
            # 送入网络
            output = net(input_list)
            est = output.permute(0, 2, 1) * input_list[:, 35:81, :]
            est_mag = torch.cat((base_list[:, 0:35, :], est), 1).permute(0, 2, 1)
            est_mag = est_mag.reshape(-1, est_mag.shape[2]).unsqueeze(0)

            p = clean_mag.mean() / output.mean()
            p_sum += p

            "方法二 一句话："
            "start"
            # # 减均值除方差
            # speech_mag_in = ((speech_mag.permute(0, 2, 1) - input_mean.cuda(CUDA_ID[0])) / (input_var.cuda(CUDA_ID[0]) + EPSILON)).permute(0, 2, 1)
            # # 送入网络
            # output = net(speech_mag_in[:, 0:21, :])
            # # revert
            # output = output * label_var[18:81].cuda(CUDA_ID[0]) + label_mean[18:81].cuda(CUDA_ID[0])
            # # TODO 测试
            # e_test = output
            # e_test = torch.exp(e_test)
            # zeros = torch.zeros(1, e_test.size()[1], 18).cuda(CUDA_ID[0])
            # e_test = torch.cat((zeros, e_test), 2)
            # # 构造预测的谱，前20个频点使用已知的，后边141个频点使用预测的
            # est_mag = torch.cat((speech_mag[:, 0:18, :], output.permute(0, 2, 1)), 1).permute(0, 2, 1)
            "end"
            # if IS_LOG:
            #     est_mag = torch.exp(est_mag)
            # 获取半波整流后的相位
            # 半波整流
            for j in range(0, len(tmp)):
                if tmp[j] < 0:
                    tmp[j] = 0
            tmp = torch.Tensor(tmp).unsqueeze(0)
            # 获取相位信息
            tmp_spec = stft.transform(tmp)
            tmp_real = tmp_spec[:, :, :, 0]
            tmp_imag = tmp_spec[:, :, :, 1]
            tmp_mag = torch.sqrt(tmp_real ** 2 + tmp_imag ** 2)
            # 恢复语音
            # TODO e_test
            # test_real = e_test.detach().cpu() * tmp_real / tmp_mag
            # test_imag = e_test.detach().cpu() * tmp_imag / tmp_mag
            # test_res = torch.stack([test_real, test_imag], 3)
            # test_res = stft.inverse(test_res)
            # sf.write('e_test.wav', test_res.numpy().squeeze(), sr)

            res_real = est_mag.detach().cpu() * tmp_real / (tmp_mag + EPSILON)
            res_imag = est_mag.detach().cpu() * tmp_imag / (tmp_mag + EPSILON)
            # 使用label的相位
            # res_real = clean_mag.detach().cpu() * tmp_real / tmp_mag
            # res_imag = clean_mag.detach().cpu() * tmp_imag / tmp_mag
            res_spec = torch.stack([res_real, res_imag], 3)
            res = stft.inverse(res_spec)
            # sf.write('/home/yangyang/userspace/data/TIMIT_low_pass/8k/generator_data/' + files[i][:-4] + '_res.wav',
            #          res.squeeze().detach().numpy(), sr)
            # 加噪
            res += torch.Tensor(noise[: res.shape[1]])
            s2 = stoi(clean.squeeze().detach().numpy(), res.squeeze().detach().numpy(), sr)
            base_stoi += s1
            res_stoi += s2
            promote_stoi += (s2 - s1)
    print(p_sum / 100)
    bar.finish()
    return base_stoi / 100, res_stoi / 100, promote_stoi / 100

# def validation_pesq_sentence(net, path):
#     net.eval()
#     net.cuda(CUDA_ID[0])
#     files = os.listdir(path)
#     files.sort()
#     sum_loss = 0
#     stft = STFT(filter_length=FILTER_LENGTH, hop_length=HOP_LENGTH)
#     bar = progressbar.ProgressBar(0, VALIDATION_DATA_NUM)
#     # bar = progressbar.ProgressBar(0, 100)
#     bar.start()
#     base_pesq = 0
#     res_pesq = 0
#     promote_pesq = 0
#     for i in range(len(files)):
#         bar.update(i)
#         # 经过滤波器的语音
#         speech, sr = sf.read(path + files[i], dtype='float32')
#         input_mean, input_var = cal_mean_var(torch.Tensor(speech[np.newaxis, :]))
#         # 对齐
#         nframe = (len(speech) - FILTER_LENGTH) // HOP_LENGTH + 1
#         size = nframe % 32
#         nframe = nframe - size
#         sample_num = FILTER_LENGTH + (nframe - 1) * HOP_LENGTH
#         speech = speech[:sample_num]
#         tmp = speech
#         sf.write(files[i], tmp, sr)
#         # speech, sr = sf.read('/home/yangyang/PycharmProjects/low_frequency_predict_high_frequency/train/REC40_R.wav')
#         # e_test label，为了计算loss
#         clean, sr = sf.read('/home/yangyang/userspace/data/TIMIT_low_pass/8k/test_label/' + files[i], dtype='float32')
#         clean = clean[:sample_num]
#         sf.write(files[i][:-4] + '_clean.wav', clean, sr)
#         # 加噪
#         # noise = np.random.random(len(clean)) * 1e-3
#         # clean += noise
#         sf.write('clean.wav', clean, sr)
#         base, sr = sf.read(path + files[i], dtype='float32')
#         base = base[:sample_num]
#         label_mean, label_var = cal_mean_var(torch.Tensor(base[np.newaxis, :]))
#         # base += noise
#         sf.write('base.wav', base, sr)
#         p1 = pesq('clean.wav', 'base.wav', is_current_file=True)
#         clean = torch.Tensor(clean).unsqueeze(0)
#         clean_spec = stft.transform(clean)
#         clean_real = clean_spec[:, :, :, 0]
#         clean_imag = clean_spec[:, :, :, 1]
#         clean_mag = torch.sqrt(clean_real ** 2 + clean_imag ** 2).cuda(CUDA_ID[0])
#         # stft
#         speech = torch.Tensor(speech).unsqueeze(0)
#         speech_spec = stft.transform(speech)
#         speech_real = speech_spec[:, :, :, 0]
#         speech_imag = speech_spec[:, :, :, 1]
#         speech_mag = torch.sqrt(speech_real ** 2 + speech_imag ** 2).cuda(CUDA_ID[0])
#         if IS_LOG:
#             speech_mag = torch.log(speech_mag + EPSILON)
#         speech_mag = speech_mag.permute(0, 2, 1).cuda(CUDA_ID[0])
#         "方法一："
#         "start"
#         speech_mag = speech_mag.squeeze()
#         input_list = []
#         for k in range(int(speech_mag.size()[1] / 32)):
#             k *= 32
#             item = speech_mag[:, k:k + 32].cpu().detach().numpy()
#             input_list.append(item)
#         input_list = torch.Tensor(np.array(input_list)).cuda(CUDA_ID[0])
#         # 归一化
#         input_list_ = (input_list - input_mean) / (input_var + EPSILON)
#         # 送入网络
#         output = net(input_list_[:, 0:21, :])
#
#         output = (output * input_var) + input_mean
#         # 构造预测的谱，前20个频点使用已知的，后边141个频点使用预测的
#         est_mag = torch.cat((input_list[:, 0:18, :], output.permute(0, 2, 1)), 1).permute(0, 2, 1)
#         est_mag = est_mag.reshape(-1, est_mag.shape[2]).unsqueeze(0)
#         "end"
#
#         "方法二："
#         "start"
#         # # 减均值除方差
#         # speech_mag_in = (speech_mag - input_mean) / (input_var + EPSILON)
#         # # 送入网络
#         # output = net(speech_mag_in[:, 0:21, :])
#         # # revert
#         # output = output * label_var + label_mean
#         # # 构造预测的谱，前20个频点使用已知的，后边141个频点使用预测的
#         # est_mag = torch.cat((speech_mag[:, 0:18, :], output.permute(0, 2, 1)), 1).permute(0, 2, 1)
#         "end"
#         if IS_LOG:
#             est_mag = torch.exp(est_mag)
#         # 获取半波整流后的相位
#         # 半波整流
#         for j in range(0, len(tmp)):
#             if tmp[j] < 0:
#                 tmp[j] = 0
#         # TODO
#         tmp = torch.Tensor(tmp).unsqueeze(0)
#         # 获取相位信息
#         tmp_spec = stft.transform(tmp)
#         tmp_real = tmp_spec[:, :, :, 0]
#         tmp_imag = tmp_spec[:, :, :, 1]
#         tmp_mag = torch.sqrt(tmp_real ** 2 + tmp_imag ** 2)
#         # 恢复语音
#
#
#         # 使用label的相位
#         res_real = est_mag.detach().cpu() * tmp_real / tmp_mag
#         res_imag = est_mag.detach().cpu() * tmp_imag / tmp_mag
#         # res_real = clean_mag.detach().cpu() * tmp_real / tmp_mag
#         # res_imag = clean_mag.detach().cpu() * tmp_imag / tmp_mag
#         res_spec = torch.stack([res_real, res_imag], 3)
#         res = stft.inverse(res_spec)
#         sf.write(files[i][:-4] + '_res.wav', res.squeeze().detach().numpy(), sr)
#         # 加噪
#         # res += torch.Tensor(noise[: res.shape[1]])
#         sf.write('res.wav', res.numpy().squeeze(), sr)
#         p2 = pesq('clean.wav', 'res.wav', is_current_file=True)
#         base_pesq += p1[0]
#         res_pesq += p2[0]
#         promote = p2[0] - p1[0]
#         promote_pesq += promote
#     bar.finish()
#     return base_pesq / 100, res_pesq / 100, promote_pesq / 100

def evaluation_discriminator(d_net):
    all = 0
    d_net.eval()
    res = []
    # files = os.listdir(VALIDATION_DATA_LABEL_PATH)
    files = os.listdir('/home/yangyang/userspace/data/TIMIT_low_pass/8k_new/generator_data/')
    train_data = loadmat('train_mean_var.mat')
    train_label_data = loadmat('train_label_mean_var.mat')
    train_mean = torch.Tensor(train_data['train_mean']).squeeze()
    train_var = torch.Tensor(train_data['train_var']).squeeze()
    train_label_mean = torch.Tensor(train_label_data['train_label_mean']).squeeze()
    train_label_var = torch.Tensor(train_label_data['train_label_var']).squeeze()
    stft = STFT(filter_length=FILTER_LENGTH, hop_length=HOP_LENGTH).cuda(CUDA_ID[0])
    time = 0
    for item in files:
        all += 1
        input_sig, sr = sf.read('/home/yangyang/userspace/data/TIMIT_low_pass/8k_new/generator_data/' + item)
        # input_sig, sr = sf.read(VALIDATION_DATA_LABEL_PATH + item)
        # frame alignment
        nframe = (len(input_sig) - FILTER_LENGTH) // HOP_LENGTH + 1
        size = nframe % 32
        nframe = nframe - size
        sample_num = FILTER_LENGTH + (nframe - 1) * HOP_LENGTH
        input_sig = input_sig[:sample_num]
        # stft
        input_spec = stft.transform(torch.Tensor(input_sig[np.newaxis, :]).cuda(CUDA_ID[0]))
        input_real = input_spec[:, :, :, 0]
        input_imag = input_spec[:, :, :, 0]
        input_mag = torch.log(torch.sqrt(input_imag ** 2 + input_real ** 2) + EPSILON)
        # input_mag = (input_mag * train_label_var.cuda(CUDA_ID[0])) + train_label_mean.cuda(CUDA_ID[0])  # 归一化
        speech_mag = input_mag.squeeze()
        input_list = []
        for k in range(int(speech_mag.size()[0] / 32)):
            k *= 32
            item = speech_mag[k:k + 32, :].cpu().detach().numpy()
            input_list.append(item)
        input_list = torch.Tensor(np.array(input_list)).cuda(CUDA_ID[0])

        res = d_net(input_list.permute(0, 2, 1))
        if res.mean() > 0.9:
            time += 1
    print(all)
    print(time)


def real_data_validation_d_net(d_net):
    label_path = '/home/yangyang/userspace/data/TIMIT_low_pass/8k_new_2x_abs/test_label/'
    d_net.eval()
    d_net.cuda(CUDA_ID[0])
    files = os.listdir(label_path)
    stft = STFT(filter_length=FILTER_LENGTH, hop_length=HOP_LENGTH)
    for i in range(len(files)):
        if files[i].endswith('.wav'):
            speech, sr = sf.read(label_path + files[i])
            nframe = (len(speech) - FILTER_LENGTH) // HOP_LENGTH + 1
            size = nframe % 32
            nframe = nframe - size
            sample_num = FILTER_LENGTH + (nframe - 1) * HOP_LENGTH
            speech = speech[:sample_num]
            # # 半波整流
            # speech[speech < 0] = 0
            # speech = np.sqrt(speech)
            speech = torch.Tensor(speech).unsqueeze(0)
            speech_spec = stft.transform(speech)
            speech_real = speech_spec[:, :, :, 0]
            speech_imag = speech_spec[:, :, :, 1]
            speech_mag = torch.sqrt(speech_real ** 2 + speech_imag ** 2).cuda(CUDA_ID[0])
            speech_mag = speech_mag.permute(0, 2, 1).cuda(CUDA_ID[0])
            "方法一 切片："
            "start"
            speech_mag = speech_mag.squeeze()
            input_list = []
            for k in range(int(speech_mag.size()[1] / 32)):
                k *= 32
                item = speech_mag[:, k:k + 32].cpu().detach().numpy()
                input_list.append(item)
            input_list = torch.Tensor(np.array(input_list)).cuda(CUDA_ID[0])
            # # 归一化
            # input_list_ = normalization(input_list)
            # input_list = input_list.permute(0, 2, 1)
            input_list = input_list.unsqueeze(3)
            # B,F,T,1
            res = d_net(input_list)
            loss_helper = LossHelper()
            loss = loss_helper.discriminator_loss_with_sigmoid(res, is_fake=False)
            print(loss.item())
            res = torch.sigmoid(res)
            # 概率
            print(res.mean())
    # z = torch.Tensor(torch.randn(64, 81, 32))
    # z.cuda(CUDA_ID[0])
    # res = d_net(z)
    # res = torch.sigmoid(res)
    # print(res.mean())


def fake_data_validation_d_net(g_net, d_net):
    path = '/home/yangyang/userspace/data/TIMIT_low_pass/8k_new_2x_abs/e_test/'
    g_net.eval()
    g_net.cuda(CUDA_ID[0])
    d_net.eval()
    d_net.cuda(CUDA_ID[0])
    files = os.listdir(path)
    files.sort()
    stft = STFT(filter_length=FILTER_LENGTH, hop_length=HOP_LENGTH)
    noise_list = np.load('noise_4_TIMIT.npy', allow_pickle=True)
    sum_seg_snr = 0
    time = 0
    for i in range(len(files)):
        if files[i].endswith('_abs.wav'):
            # 经过滤波器的语音
            # input
            speech, sr = sf.read(path + files[i])
            # 对齐
            nframe = (len(speech) - FILTER_LENGTH) // HOP_LENGTH + 1
            size = nframe % 32
            nframe = nframe - size
            sample_num = FILTER_LENGTH + (nframe - 1) * HOP_LENGTH
            speech = speech[:sample_num]
            speech[speech < 0] = 0
            speech = np.sqrt(speech)
            # e_test label，为了计算loss
            clean, sr = sf.read('/home/yangyang/userspace/data/TIMIT_low_pass/8k_new_2x_abs/test_label/' + files[i][:-8] + '.wav')
            clean = clean[:sample_num]
            # 加噪
            noise = noise_list[time]
            time += 1
            clean += noise
            base, sr = sf.read(path + files[i][:-8] + '.wav')
            base = base[:sample_num]
            tmp = base
            # input
            base += noise
            base_mag = stft.spec_transform(stft.transform(torch.Tensor(base[np.newaxis, :]))).squeeze()
            base_mag = base_mag.permute(1, 0)
            clean = torch.Tensor(clean).unsqueeze(0)
            clean_spec = stft.transform(clean)
            clean_real = clean_spec[:, :, :, 0]
            clean_imag = clean_spec[:, :, :, 1]
            clean_mag = torch.sqrt(clean_real ** 2 + clean_imag ** 2).cuda(CUDA_ID[0]).squeeze()
            # stft
            speech = torch.Tensor(speech).unsqueeze(0)
            speech_spec = stft.transform(speech)
            speech_real = speech_spec[:, :, :, 0]
            speech_imag = speech_spec[:, :, :, 1]
            speech_mag = torch.sqrt(speech_real ** 2 + speech_imag ** 2).cuda(CUDA_ID[0])
            speech_mag = speech_mag.permute(0, 2, 1).cuda(CUDA_ID[0])
            "方法一 切片："
            "start"
            # 切片
            speech_mag = speech_mag.squeeze()
            # speech_mag = torch.cat((base_mag[0:18, :].cuda(CUDA_ID[0]), speech_mag[18:81, :]), 0)
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
            # 送入网络
            # input_list = input_list.unsqueeze(3)
            output = g_net(input_list)
            # 构造预测的谱，前20个频点使用已知的，后边141个频点使用预测的
            est_mag = torch.cat((base_list[:, 0:35, :], output.permute(0, 2, 1)), 1).permute(0, 2, 1)
            "end"
            loss_helper = LossHelper()
            # seg_snr = loss_helper.seg_SNR(est_mag, clean_list, batch_first=True).mean()
            # sum_seg_snr += seg_snr.item()
            # print('seg snr : ' + str(seg_snr))
            "end"
            # B,F,T,1
            d_net_output = d_net(est_mag.permute(0, 2, 1).unsqueeze(3))
            d_net_output = torch.sigmoid(d_net_output)
            print(d_net_output.mean().item())
            loss = loss_helper.generator_loss_with_sigmoid(d_net_output)
            print('loss : ' + str(loss.item()))
            # d_net_output = torch.sigmoid(d_net_output)
            # print('d_net output :' + str(d_net_output.mean().item()))
    # return sum_seg_snr / 100


def cal_metrics(g_net):
    g_net.eval()
    g_net.cuda(CUDA_ID[0])
    files = os.listdir('/home/yangyang/userspace/data/TIMIT_low_pass/8k_new_2x_abs/e_test/')
    files.sort()
    stft = STFT(filter_length=FILTER_LENGTH, hop_length=HOP_LENGTH)
    noise_list = np.load('noise_4_TIMIT.npy', allow_pickle=True)
    sum_seg_snr = 0
    sum_half_lsd = 0
    sum_full_lsd = 0
    time = 0
    loss_helper = LossHelper()
    for i in range(len(files)):
        if files[i].endswith('_abs.wav'):
        # 经过滤波器的语音
            # input
            speech, sr = sf.read('/home/yangyang/userspace/data/TIMIT_low_pass/8k_new_2x_abs/e_test/' + files[i])
            # 对齐
            nframe = (len(speech) - FILTER_LENGTH) // HOP_LENGTH + 1
            size = nframe % 32
            nframe = nframe - size
            sample_num = FILTER_LENGTH + (nframe - 1) * HOP_LENGTH
            speech = speech[:sample_num]
            # speech[speech < 0] = 0
            # speech = np.sqrt(speech)
            # e_test label，为了计算loss
            clean, sr = sf.read('/home/yangyang/userspace/data/TIMIT_low_pass/8k_new_2x_abs/test_label/' + files[i][:-8] + '.wav')
            clean = clean[:sample_num]
            # 加噪
            noise = noise_list[time]
            time += 1
            clean += noise
            base, sr = sf.read('/home/yangyang/userspace/data/TIMIT_low_pass/8k_new_2x_abs/e_test/' + files[i][:-8] + '.wav')
            base = base[:sample_num]
            tmp = base
            # input
            base += noise
            base_mag = stft.spec_transform(stft.transform(torch.Tensor(base[np.newaxis, :]))).squeeze()
            base_mag = base_mag.permute(1, 0)
            clean = torch.Tensor(clean).unsqueeze(0)
            clean_spec = stft.transform(clean)
            clean_real = clean_spec[:, :, :, 0]
            clean_imag = clean_spec[:, :, :, 1]
            clean_mag = torch.sqrt(clean_real ** 2 + clean_imag ** 2).cuda(CUDA_ID[0]).squeeze()
            # stft
            speech = torch.Tensor(speech).unsqueeze(0)
            speech_spec = stft.transform(speech)
            speech_real = speech_spec[:, :, :, 0]
            speech_imag = speech_spec[:, :, :, 1]
            speech_mag = torch.sqrt(speech_real ** 2 + speech_imag ** 2).cuda(CUDA_ID[0])
            speech_mag = speech_mag.permute(0, 2, 1).cuda(CUDA_ID[0])
            "方法一 切片："
            "start"
            # 切片
            speech_mag = speech_mag.squeeze()
            # speech_mag = torch.cat((base_mag[0:18, :].cuda(CUDA_ID[0]), speech_mag[18:81, :]), 0)
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
            # 送入网络
            output = g_net(input_list)
            est = output.permute(0, 2, 1) * input_list[:, 35:81, :]
            # 构造预测的谱，前20个频点使用已知的，后边141个频点使用预测的
            est_mag = torch.cat((base_list[:, 0:35, :], est), 1).permute(0, 2, 1)
            loss = loss_helper.MSE_loss(est_mag.reshape(-1, 81), clean_mag)
            half_lsd = loss_helper.mertics_LSD(est_mag, clean_list, is_full=False)
            "end"
            seg_snr = loss_helper.seg_SNR(est_mag, clean_list, batch_first=True).mean()
            full_lsd = loss_helper.mertics_LSD(est_mag, clean_list, is_full=True)
            sum_full_lsd += full_lsd.item()
            sum_half_lsd += half_lsd.item()
            sum_seg_snr += seg_snr.item()
            "end"
    return sum_seg_snr / 100, sum_half_lsd / 100, sum_full_lsd / 100


def cal_all_metrics():
    if os.path.exists(MODEL_STORE + 'all_model_metrics/'):
        shutil.rmtree(MODEL_STORE + 'all_model_metrics/')
    os.mkdir(MODEL_STORE + 'all_model_metrics/')
    log_store = MODEL_STORE + 'all_model_metrics/'
    writer = SummaryWriter(log_store)
    for i in range(0, 85):
        g_net = GeneratorNet()
        res = resume_model(g_net, MODEL_STORE + 'TIMIT_train_learn_speech_with_sn_with_noise_sigmoid_10_mse_10/g_model_' + str(i * 2000) + '.pkl')
        seg_snr, half_lsd, full_lsd = cal_metrics(g_net)
        writer.add_scalar('Metrics/snr', seg_snr, i)
        writer.add_scalar('Metrics/half_lsd', half_lsd, i)
        writer.add_scalar('Metrics/full_lsd', full_lsd, i)

def cal_loss_4_test(net, output_path):
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
    sum_loss = 0
    feature_creator = FeatureCreator()
    noise_list = np.load('noise_4_TIMIT_new.npy', allow_pickle=True)
    loss_helper = LossHelper()
    if NEED_NORM:
        if IS_LOG:
            train_mean = feature_creator.log_train_mean
            train_var = feature_creator.log_train_var
        else:
            train_mean = feature_creator.train_mean
            train_var = feature_creator.train_var
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
            sf.write(output_path + files[i], speech, sr)
            # tmp 为半波整流之后获取相位用
            base = speech.copy()
            tmp = speech.copy()
            tmp_low = speech.copy()
            sf.write('base_cat.wav', base, sr)
            base_mag = stft.spec_transform(stft.transform(torch.Tensor(base[np.newaxis, :]))).squeeze()
            base_mag = base_mag.permute(1, 0)
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
            # noise = noise_list[i][:sample_num]
            # add noise for calculate real pesq
            # clean += noise
            # for calculate seg_snr
            clean_frame = vec2frame(clean, 32, 8)
            sf.write(output_path + files[i][:-4] + '_clean.wav', clean, sr)
            sf.write('clean_cat.wav', clean, sr)
            # 低频部分的语音，最后cat在一起用
            # base, sr = sf.read(VALIDATION_DATA_PATH + files[i])
            # sf.write(output_path + files[i][:-4] + '_base.wav', base, sr)
            # base = base[:sample_num]
            # base += noise

            p1 = pesq('clean_cat.wav', 'base_cat.wav', is_current_file=True)
            clean = torch.Tensor(clean).unsqueeze(0)
            clean_spec = stft.transform(clean)
            clean_real = clean_spec[:, :, :, 0]
            clean_imag = clean_spec[:, :, :, 1]
            clean_mag = torch.sqrt(clean_real ** 2 + clean_imag ** 2).cuda(CUDA_ID[0]).squeeze()
            # 切片
            speech_mag = speech_mag.squeeze()
            # input_list = speech_mag.reshape(-1, speech_mag.size()[1] // 32, 32)
            # base_list = base_mag.reshape(-1, speech_mag.size()[1] // 32, 32)
            # clean_list = clean_mag.reshape(-1, speech_mag.size()[1] // 32, 32)
            base_list = []
            clean_list = []
            input_list = []
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
            est_speech = net(input_list[:, :129, :])
            if NEED_NORM:
                est_speech = est_speech * train_var[116:] + train_mean[116:]
            if IS_LOG:
                est_speech = torch.exp(est_speech)
            # savemat(files[i][:-4] + '.mat', {'est': est_speech.reshape(-1, 71).detach().cpu().numpy(), 'clean_mag': clean_mag[:, 116:].detach().cpu().numpy()})
            est_mag = torch.cat((base_list.permute(0, 2, 1)[:, :, :116], est_speech), 2)
            loss = loss_helper.MSE_loss(est_mag, clean_list)
            sum_loss += loss.item()
            # savemat('tmp.mat', {'est': est_speech.reshape(-1, 71).detach().cpu().numpy(), 'clean_mag': clean_mag[:, 116:].log().detach().cpu().numpy()})
            # savemat('tmp.mat', {'est_mag': est_mag.reshape(-1, 129).detach().cpu().numpy(), 'clean_mag': clean_mag.detach().cpu().numpy()})
            # est_mag = torch.cat((base_list.permute(0, 2, 1)[:, :, :116], clean_list[:, :, 116:]), 2)
            # half_lsd = loss_helper.mertics_LSD(est_mag, clean_list, is_full=False)
            # full_lsd = loss_helper.mertics_LSD(est_mag, clean_list, is_full=True)
            # # 合并
            # est_mag = est_mag.reshape(-1, est_mag.shape[2]).unsqueeze(0)
            # "end"
            #
            # sum_full_lsd += full_lsd.item()
            # sum_half_lsd += half_lsd.item()
            # # 获取低频相位信息
            # tmp_low_spec = stft.transform(torch.Tensor(tmp_low[np.newaxis, :]))
            # tmp_low_real = tmp_low_spec[:, :, :, 0]
            # tmp_low_imag = tmp_low_spec[:, :, :, 1]
            # tmp_low_mag = torch.sqrt(tmp_low_real ** 2 + tmp_low_imag ** 2).squeeze()
            # "end"
            #
            # # 获取高频相位信息
            # tmp[tmp < 0] = 0
            # tmp = torch.Tensor(tmp).unsqueeze(0)
            # tmp_spec = stft.transform(tmp)
            # tmp_real = tmp_spec[:, :, :, 0]
            # tmp_imag = tmp_spec[:, :, :, 1]
            # tmp_mag = torch.sqrt(tmp_real ** 2 + tmp_imag ** 2)
            # # 低频使用原始相位，高频使用半波整流之后的语音的相位
            # tmp_mag = torch.cat((tmp_low_mag[:, :129], tmp_mag.squeeze()[:, 129:]), 1)
            # tmp_real = torch.cat((tmp_low_real.squeeze()[:, :129], tmp_real.squeeze()[:, 129:]), 1)
            # tmp_imag = torch.cat((tmp_low_imag.squeeze()[:, :129], tmp_imag.squeeze()[:, 129:]), 1)
            # # 纯净语音的相位
            # # tmp_mag = clean_mag.detach().cpu()
            # # tmp_real = clean_real.detach().cpu()
            # # tmp_imag = clean_imag.detach().cpu()
            # # TODO e_test
            # # test_real = e_test.detach().cpu() * tmp_real / tmp_mag
            # # test_imag = e_test.detach().cpu() * tmp_imag / tmp_mag
            # # test_res = torch.stack([test_real, test_imag], 3)
            # # test_res = stft.inverse(test_res)
            # # sf.write('e_test.wav', test_res.numpy().squeeze(), sr)
            # # 恢复语音
            # res_real = est_mag.detach().cpu() * tmp_real / tmp_mag
            # res_imag = est_mag.detach().cpu() * tmp_imag / tmp_mag
            # res_spec = torch.stack([res_real, res_imag], 3)
            # res = stft.inverse(res_spec)
            # sf.write(output_path + files[i][:-4] + '_res.wav', res.squeeze().detach().numpy(), sr)
            # # 加噪
            # # res += torch.Tensor(noise[: res.shape[1]])
            # sf.write('res_cat.wav', res.numpy().squeeze(), sr)
            # p2 = pesq('clean_cat.wav', 'res_cat.wav', is_current_file=True)
            # res_frame = vec2frame(res, 32, 8)
            # loss_helper = LossHelper()
            # seg_snr = loss_helper.seg_SNR_(torch.Tensor(res_frame), torch.Tensor(clean_frame))
            #
            # sum_seg_snr += seg_snr.item()
            # base_pesq += p1[0]
            # res_pesq += p2[0]
            # promote = p2[0] - p1[0]
            # promote_pesq += promote
    bar.finish()
    return sum_loss / 100

if __name__ == '__main__':
    # step = 2000
    # for i in range(1, 160):
    #     d_net = DiscriminatorNet()
    #     print(i)
    #     res = resume_discriminator_model(d_net, MODEL_STORE + 'train_learn_speech/d_model_' + str(step * i) + '.pkl')
    #     net = GeneratorNet()
    #     res = resume_model(net, MODEL_STORE + 'train_learn_speech/g_model_' + str(step * i) + '.pkl')
    #     seg_snr = fake_data_validation_d_net(net, d_net)
    #     print(seg_snr)
    #     print("========================================================")
    # cal_all_metrics()
    # d_net = DiscriminatorNet()
    # res = resume_discriminator_model(d_net, MODEL_STORE + 'TIMIT_abs_train_learn_speech_with_sn_without_noise_sigmoid_10_mse_10_lr_1e-4/d_model_18000.pkl')
    g_net = GeneratorNet()
    # res = resume_model(g_net, MODEL_STORE + 'IEEE_pre_train_g_learn_speech_with_noise_sigmoid_10/model_16000.pkl')
    # res = resume_model(g_net, MODEL_STORE + 'TIMIT_abs_train_learn_speech_with_sn_without_noise_sigmoid_10_mse_10_lr_1e-4/g_model_18000.pkl')
    res = resume_model(g_net, MODEL_STORE + 'TIMIT_16k_train_learn_speech_with_sn_lsd_no_pre_train_d_in_norm/g_model_18000.pkl')
    # real_data_validation_d_net(d_net)
    # fake_data_validation_d_net(g_net, d_net)
    # seg_snr, half_lsd, full_lsd = cal_metrics(g_net)
    # print('seg snr : ' + str(seg_snr))
    # print('half lsd : ' + str(half_lsd))
    # print('full lsd : ' + str(full_lsd))
    # d_net = DiscriminatorNet()
    # res1 = resume_discriminator_model(d_net, MODEL_STORE + 'train_learn_speech_with_sn_with_disc/d_model_348000.pkl')
    # seg_snr = fake_data_validation_d_net(g_net, g_net)
    # print(seg_snr)
    # base_stoi, res_stoi, promote_stoi = validation_stoi(g_net)
    # print('base stoi : ' + str(base_stoi))
    # print('res stoi : ' + str(res_stoi))
    # print('promote stoi : ' + str(promote_stoi))
    print('====================================')
    output_path = '/home/yangyang/userspace/data/TIMIT_low_pass/8k_new_2x/TIMIT_16k_train_learn_speech_with_sn_lsd_no_pre_train_d_in_norm_18k/'
    if os.path.exists(output_path):
        shutil.rmtree(output_path)
    os.mkdir(output_path)
    base_pesq, res_pesq, promote_pesq, seg_snr, half_lsd, full_lsd = validation_pesq(g_net, output_path)
    # loss = cal_loss_4_test(g_net, output_path)
    # print('loss : ' + str(loss))
    print('base pesq : ' + str(base_pesq))
    print('res pesq : ' + str(res_pesq))
    print('seg snr : ' + str(seg_snr))
    print('promote pesq : ' + str(promote_pesq))
    print('half lsd : ' + str(half_lsd))
    print('full lsd : ' + str(full_lsd))





    #g_model_260000
    # g_net = GeneratorNet()
    # g_net.cuda(CUDA_ID[0])
    # net = DiscriminatorNet()
    # res = resume_model(net, MODEL_STORE + 'd_model_pre_train.pkl')
    # net.cuda(CUDA_ID[0])
    # evaluation_discriminator(net)
