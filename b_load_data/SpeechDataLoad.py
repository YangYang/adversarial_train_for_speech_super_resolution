from utils.label_set import LabelHelper
import torch
import torch.nn as nn
import numpy as np
from scipy.io import loadmat
from utils.util import context_window
from utils.stft_istft import STFT
from utils.util import normalization
import scipy.io as sio
import soundfile as sf
from config import *
from torch.utils.data import DataLoader, Dataset


class SpeechDataLoader(object):
    def __init__(self, data_set, batch_size, is_shuffle=True, num_workers=NUM_WORKERS):
        """
        初始化一个系统的Dataloader，只重写他的collate_fn方法
        :param data_set: 送入网络的data,dataset对象
        :param batch_size: 每次送入网络的data的数量，即多少句话
        :param is_shuffle: 是否打乱送入网络
        :param num_workers: dataloader多线程工作数，一般我们取0
        """
        self.data_loader = DataLoader(dataset=data_set,
                                      batch_size=batch_size,
                                      shuffle=is_shuffle,
                                      num_workers=num_workers,
                                      collate_fn=self.collate_fn,
                                      drop_last=True)

    # 静态方法，由类和对象调用
    # 该函数返回对数据的处理，返回target,load_data
    @staticmethod
    def collate_fn(batch):
        """
        将每个batch中的数据pad成一样长，采取补零操作
        切记为@staticmethod方法
        :param batch: input和label的list
        :return:input、label和真实帧长 的list
        """
        label_list = []
        speech_list = []
        train_wav_list = []
        for item in batch:
            # (T,F)
            speech_list.append(item[0])
            label_list.append(item[1])
            train_wav_list.append(item[2])
            # 储存每句话的真实帧长，时域信息，用于计算loss
        # 把mix、speech和noise pad成一样长度
        speech_list = nn.utils.rnn.pad_sequence(speech_list)
        label_list = nn.utils.rnn.pad_sequence(label_list)
        train_wav_list = nn.utils.rnn.pad_sequence(train_wav_list)

        # data_list = (B,in_c,T,F)
        # target_list = (B,T,F)
        return BatchInfo(speech_list, label_list, train_wav_list)

    def get_data_loader(self):
        """
        获取Dataloader
        :return: dataloader对象
        """
        return self.data_loader


class SpeechDataset(Dataset):

    def __getitem__(self, index):
        input = self.train[index * (FILTER_LENGTH + (HOP_LENGTH * 31)): (index + 1) * (FILTER_LENGTH + (HOP_LENGTH * 31))]
        label = self.train_label[index * (FILTER_LENGTH + (HOP_LENGTH * 31)): (index + 1) * (FILTER_LENGTH + (HOP_LENGTH * 31))]
        train_wav = self.train_wav[index * (FILTER_LENGTH + (HOP_LENGTH * 31)): (index + 1) * (FILTER_LENGTH + (HOP_LENGTH * 31))]
        return torch.Tensor(input), torch.Tensor(label), torch.Tensor(train_wav)

    def __len__(self):
        return self.frame_len // 32

    def __init__(self, file_dir, label_file_dir):
        stft = STFT(filter_length=FILTER_LENGTH, hop_length=HOP_LENGTH)
        # TODO  初始化变量
        self.train_file_dir = file_dir
        self.train_label_file_dir = label_file_dir
        self.train, sr = sf.read(file_dir)
        self.train_wav, sr = sf.read(self.train_file_dir)
        self.train_label, sr = sf.read(label_file_dir)
        self.frame_len = (len(self.train) - FILTER_LENGTH) // HOP_LENGTH + 1
        # # input的均值和方差
        # train_data = loadmat('train_mean_var.mat')
        # train_label_data = loadmat('train_label_mean_var.mat')
        # self.train_mean = torch.Tensor(train_data['train_mean']).squeeze()
        # self.train_var = torch.Tensor(train_data['train_var']).squeeze()
        # # label的均值和方差
        # self.train_label_mean = torch.Tensor(train_label_data['train_label_mean']).squeeze()
        # self.train_label_var = torch.Tensor(train_label_data['train_label_var']).squeeze()


class BatchInfo(object):

    def __init__(self, speech, label, train_wav):
        self.speech = speech
        self.label = label
        self.train_wav = train_wav


class FeatureCreator(nn.Module):

    def __init__(self):
        super(FeatureCreator, self).__init__()
        self.stft = STFT(FILTER_LENGTH, HOP_LENGTH).cuda(CUDA_ID[0])
        self.label_helper = LabelHelper().cuda(CUDA_ID[0])
        if NEED_NORM:
            if IS_LOG:
                self.train_mean = torch.Tensor(np.load(TRAIN_PARAM_PATH + 'train_log_mean.npy')).cuda(CUDA_ID[0])
                self.train_var = torch.Tensor(np.load(TRAIN_PARAM_PATH + 'train_log_var.npy')).cuda(CUDA_ID[0])
                self.train_label_mean = torch.Tensor(np.load(TRAIN_PARAM_PATH + 'train_label_log_mean.npy')).cuda(CUDA_ID[0])
                self.train_label_var = torch.Tensor(np.load(TRAIN_PARAM_PATH + 'train_label_log_var.npy')).cuda(CUDA_ID[0])
            else:
                self.train_mean = torch.Tensor(np.load(TRAIN_PARAM_PATH + 'train_mean.npy')).cuda(CUDA_ID[0])
                self.train_var = torch.Tensor(np.load(TRAIN_PARAM_PATH + 'train_var.npy')).cuda(CUDA_ID[0])
                self.train_label_mean = torch.Tensor(np.load(TRAIN_PARAM_PATH + 'train_label_mean.npy')).cuda(CUDA_ID[0])
                self.train_label_var = torch.Tensor(np.load(TRAIN_PARAM_PATH + 'train_label_var.npy')).cuda(CUDA_ID[0])

    def forward(self, batch_info):
        # label 63
        # 计算幅度谱
        g_input = self.stft.spec_transform(self.stft.transform(batch_info.train_wav.transpose(1, 0).cuda(CUDA_ID[0])))
        # label mag
        speech_spec = self.stft.transform(batch_info.label.transpose(1, 0).cuda(CUDA_ID[0]))
        g_label = self.label_helper(speech_spec)
        sio.savemat('data.mat', {'input': g_input.detach().cpu().numpy(),
                                 'label': g_label.detach().cpu().numpy()})
        if IS_LOG:
            g_input = torch.log(g_input + EPSILON)
            g_label = torch.log(g_label + EPSILON)
        if NEED_NORM:
            if IS_LOG:
                g_input = (g_input - self.train_mean) / (self.train_var + torch.Tensor(np.array(EPSILON)).cuda(CUDA_ID[0]))
                g_label = (g_label - self.train_label_mean) / (self.train_label_var + torch.Tensor(np.array(EPSILON)).cuda(CUDA_ID[0]))
        # permute (N, C, L)
        return g_input, g_label

    def revert_norm(self, est_speech, label):
        if SAMPLING_RATE == 8000:
            est_speech_revert = est_speech * self.train_label_var[58:] + self.train_label_mean[58:]
        elif SAMPLING_RATE == 16000:
            est_speech_revert = est_speech * self.train_label_var[116:] + self.train_label_mean[116:]
        label_revert = label * self.train_label_var + self.train_label_mean
        return est_speech_revert, label_revert
