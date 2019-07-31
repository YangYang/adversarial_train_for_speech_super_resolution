from a_net.module_with_SN_8k_2x import GeneratorNet, DiscriminatorNet
import torch
import numpy as np
from config import *
import soundfile as sf
from utils.stft_istft import STFT
import scipy.io as sio
import librosa
"e_test for net"
"start"
# # B, F, T
# g_input = torch.Tensor(np.random.randn(64, 65, 32))
# g_net = GeneratorNet()
# d_net = DiscriminatorNet()
# g_output = g_net(g_input)
# # B, T, F
# d_input = torch.cat((g_output.permute(0, 2, 1), g_input), 1)
# d_input = d_input.unsqueeze(3)
# # d_input : B, F, T, 1
# d_output = d_net(d_input)
# print(d_output)
"end"

"create mean and var"
"start"
# sig, sr = sf.read(TRAIN_DATA_PATH)
# stft = STFT(filter_length=FILTER_LENGTH, hop_length=HOP_LENGTH)
# sig_log_mag = stft.spec_transform(stft.transform(torch.Tensor(sig[np.newaxis, ])))
# # sig_log_mag = torch.log(sig_mag + EPSILON)
# mean = sig_log_mag.squeeze().mean(0)
# var = sig_log_mag.squeeze().var(0)
# np.save('train_mean', mean)
# np.save('train_var', var)
# train_mean = np.load('train_mean.npy', allow_pickle=True)
# train_var = np.load('train_var.npy', allow_pickle=True)
"end"

# train_param = sio.loadmat('train_param')
# print(train_param)

# files = os.listdir(VALIDATION_DATA_PATH)
# files.sort()
# noise_list = []
# for item in files:
#     sig, sr = sf.read(VALIDATION_DATA_PATH + item)
#     noise = np.random.random(sig.size) * 1e-3
#     noise_list.append(noise)
# np.save('noise_4_TIMIT_new.npy', np.array(noise_list))
def cal_train_label_mean():
    sig, sr = sf.read(TRAIN_LABEL_PATH)
    stft = STFT(filter_length=FILTER_LENGTH, hop_length=HOP_LENGTH)
    label_mag = stft.spec_transform(stft.transform(torch.Tensor(sig[np.newaxis, :]))).squeeze()
    label_mag_mean = label_mag.log().mean(0)
    label_mag_var = label_mag.log().var(0)
    np.save(TRAIN_PARAM_PATH + 'train_label_log_mean.npy', label_mag_mean.detach().numpy())
    np.save(TRAIN_PARAM_PATH + 'train_label_log_var.npy', label_mag_var.detach().numpy())
    print(label_mag_mean.size())
    print(label_mag_var.size())

if __name__ == '__main__':
    # cal_train_label_mean()
    res = np.load(TRAIN_PARAM_PATH + 'train_log_mean.npy')
    print(res.shape)
    print(res)
    stft = STFT(filter_length=FILTER_LENGTH, hop_length=HOP_LENGTH)
    input, sr = sf.read('/home/yangyang/userspace/data/TIMIT_low_pass/8k_new_2x/test_label/test_dr1_faks0_sa1.wav')
    clean, sr = sf.read('/home/yangyang/userspace/data/TIMIT_low_pass/8k_new_2x/test_label/test_dr1_faks0_sa1.wav')
    tmp = clean.copy()
    # tmp[tmp < 0] = 0
    tmp_spec = librosa.stft(tmp, 256, 64)
    tmp_real = tmp_spec.real
    tmp_imag = tmp_spec.imag
    tmp_mag = abs(tmp_spec)
    # tmp_spec = stft.transform(torch.Tensor(tmp[np.newaxis, :]))
    # tmp_real = tmp_spec[:, :, :, 0]
    # tmp_imag = tmp_spec[:, :, :, 1]
    # tmp_mag = (tmp_real ** 2 + tmp_imag ** 2).sqrt()
    # tmp_low = input.copy()
    input_spec = librosa.stft(input, 256, 64)

    # input_spec = stft.transform(torch.Tensor(input[np.newaxis, :]))
    # input_real = input_spec[:, :, :, 0]
    # input_imag = input_spec[:, :, :, 1]
    # input_mag = (input_real ** 2 + input_imag ** 2).sqrt()
    input_mag = abs(input_spec)
    clean_spec = librosa.stft(clean, 256, 64)
    clean_mag = abs(clean_spec)
    # clean_spec = stft.transform(torch.Tensor(clean[np.newaxis, :]))
    # clean_real = clean_spec[:, :, :, 0]
    # clean_imag = clean_spec[:, :, :, 1]
    # clean_mag = (clean_real ** 2 + clean_imag ** 2).sqrt()

    est_mag = np.concatenate((input_mag[:, :58], clean_mag[:, 58:]), 1)
    tmp_angle = tmp_real.squeeze() + 1j * tmp_imag.squeeze()
    res_spec = est_mag.squeeze() * np.exp(1j * np.angle(tmp_angle))
    # res_spec = torch.Tensor(np.stack([res.real, res.imag], 2)[np.newaxis, :, :, :])
    "方法二"
    # res_real = est_mag * tmp_real / tmp_mag
    # res_imag = est_mag * tmp_imag / tmp_mag
    # res_spec = torch.stack([res_real, res_imag], 3)
    res = librosa.istft(res_spec)
    # res = stft.inverse(res_spec).squeeze()
    sf.write('tmp5.wav', res, sr)



