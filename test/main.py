from a_net.module_with_SN_8k_2x import GeneratorNet, DiscriminatorNet
import torch
import numpy as np
from config import *
import soundfile as sf
from utils.stft_istft import STFT
import scipy.io as sio
"test for net"
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
sig, sr = sf.read(TRAIN_DATA_PATH)
stft = STFT(filter_length=FILTER_LENGTH, hop_length=HOP_LENGTH)
sig_log_mag = stft.spec_transform(stft.transform(torch.Tensor(sig[np.newaxis, ])))
# sig_log_mag = torch.log(sig_mag + EPSILON)
mean = sig_log_mag.squeeze().mean(0)
var = sig_log_mag.squeeze().var(0)
np.save('train_mean', mean)
np.save('train_var', var)
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
