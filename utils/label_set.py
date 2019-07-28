import torch
import torch.nn as nn
from config import *
from utils.stft_istft import STFT


class LabelHelper(nn.Module):

    def __init__(self):
        super(LabelHelper, self).__init__()
        self.stft = STFT(FILTER_LENGTH, HOP_LENGTH)

    def forward(self, speech_spec):
        return self.cal_speech_mag(speech_spec)

    def cal_IRM(self, speech_spec, noise_spec):
        noise_real = noise_spec[:, :, :, 0]
        noise_imag = noise_spec[:, :, :, 1]
        speech_real = speech_spec[:, :, :, 0]
        speech_imag = speech_spec[:, :, :, 1]
        speech_mag = torch.sqrt(speech_real.pow(2) + speech_imag.pow(2)).squeeze()
        noise_mag = torch.sqrt(noise_real.pow(2) + noise_imag.pow(2)).squeeze()
        IRM = torch.sqrt(speech_mag ** 2 / (speech_mag ** 2 + noise_mag ** 2 + EPSILON))
        # (B,T,F)
        return IRM

    def cal_speech_mag(self, speech_spec):
        speech_real = speech_spec[:, :, :, 0]
        speech_imag = speech_spec[:, :, :, 1]
        return torch.sqrt(speech_real ** 2 + speech_imag ** 2)

    def cal_label(self, mix_mag):
        mix_mag = mix_mag.unsqueeze(1)
        label = self.model(mix_mag)
        return label

    def cal_PSM(self, s, n):
        y = s + n
        s_real = s[:, :, :, 0]
        s_imag = s[:, :, :, 1]
        y_real = y[:, :, :, 0]
        y_imag = y[:, :, :, 1]
        return ((s_real * y_real + s_imag * y_imag) / (EPSILON + y_real ** 2 + y_imag ** 2)).clamp(0, 1).squeeze()
