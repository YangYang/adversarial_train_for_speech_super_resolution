import torch
from torch.nn.utils.rnn import pad_sequence
import torch.nn as nn
from config import *


class LossHelper(object):

    @staticmethod
    def mse_loss(est, label, nframes):
        """
        计算真实的MSE
        :param est: 网络输出
        :param label: label
        :param nframes: 每个batch中的真实帧长
        :return:loss
        """
        with torch.no_grad():
            mask_for_loss_list = []
            # 制作掩码
            for frame_num in nframes:
                mask_for_loss_list.append(torch.ones(frame_num, label.size()[2], dtype=torch.float32))
            # input: list of tensor
            # output: B T *
            mask_for_loss = pad_sequence(mask_for_loss_list, batch_first=True).cuda(CUDA_ID[0])
        # 使用掩码计算真实值
        masked_est = est * mask_for_loss
        masked_label = label * mask_for_loss
        loss = ((masked_est - masked_label) ** 2).sum() / mask_for_loss.sum()
        return loss

    @staticmethod
    def cross_entropy_loss(input, target):
        loss_fn = nn.CrossEntropyLoss()
        loss = loss_fn(input, target)
        return loss

    @staticmethod
    def MSE_loss(est, label):
        loss_fn = torch.nn.MSELoss()
        loss = loss_fn(est, label)
        return loss

    @staticmethod
    def KLDiv_loss(est, label):
        loss_fn = torch.nn.KLDivLoss()
        loss = loss_fn(est, label)
        return loss

    @staticmethod
    def LSD_loss(est, label):
        # loss = (torch.sum(torch.sqrt(torch.sum(torch.pow(label - est, 2), 2) / est.size()[2]), 1) / est.size()[1]).mean()
        # loss = (torch.log(label + EPSILON) - torch.log(est + EPSILON)).pow(2).mean(2).sqrt().mean(1).mean()
        loss = (label - est).pow(2).mean(2).sqrt().mean(1).mean()
        # loss = torch.sum(torch.sqrt(torch.sum(torch.pow(label - est, 2), 1) / est.size()[1])) / est.size()[0]
        return loss

    @staticmethod
    def LSD_loss_with_sigmoid(est, label):
        est = torch.sigmoid(est)
        loss = (label - est).pow(2).mean(2).mean(1).mean()
        # loss = torch.sum(torch.sqrt(torch.sum(torch.pow(label - est, 2), 1) / est.size()[1])) / est.size()[0]
        return loss

    @staticmethod
    def discriminator_loss(logits_fake, is_fake):
        bce_loss = nn.BCELoss()
        if is_fake:
            label = torch.zeros(logits_fake.size()[0], 1).cuda(CUDA_ID[0])
        else:
            label = torch.ones(logits_fake.size()[0], 1).cuda(CUDA_ID[0]) - 0.1
        loss = bce_loss(logits_fake, label)
        return loss

    @staticmethod
    def discriminator_loss_with_sigmoid(logits_fake, is_fake):
        bce_loss = nn.BCEWithLogitsLoss()
        if is_fake:
            label = torch.zeros(logits_fake.size()[0], 1).cuda(CUDA_ID[0])
        else:
            label = torch.ones(logits_fake.size()[0], 1).cuda(CUDA_ID[0]) - 0.1
        loss = bce_loss(logits_fake, label)
        return loss

    @staticmethod
    def generator_loss(logits_fake):
        """
        训练生成器时候的loss，即使用真label和判别器根据假input生成的logits_fake，使其接近从而更新生成器
        :param logits_fake:判别器根据假input作出的判断
        :return:
        """
        bce_loss = nn.BCELoss()
        size = logits_fake.shape[0]
        true_labels = torch.ones(size, 1).cuda(CUDA_ID[0])
        loss = bce_loss(logits_fake, true_labels)
        return loss

    @staticmethod
    def generator_loss_with_sigmoid(logits_fake):
        """
        训练生成器时候的loss，即使用真label和判别器根据假input生成的logits_fake，使其接近从而更新生成器
        :param logits_fake:判别器根据假input作出的判断
        :return:
        """
        bce_loss = nn.BCEWithLogitsLoss()
        size = logits_fake.shape[0]
        true_labels = torch.ones(size, 1).cuda(CUDA_ID[0])
        loss = bce_loss(logits_fake, true_labels)
        return loss

    @staticmethod
    def seg_SNR(est, real, batch_first=True):
        """
        预测的语音和真实语音的segSNR
        :param est:(B,T,F) 预测的语音
        :param real:(B,T,F) 真实的语音
        :return:分段SNR
        """
        if batch_first:
            seg_sng = torch.sum(10.0 * torch.log10(torch.sum(torch.pow(real, 2), 2) / torch.sum(torch.pow(real - est, 2), 2)), 1) / real.size()[1]
        else:
            seg_sng = torch.sum(10.0 * torch.log10(torch.sum(torch.pow(real, 2), 1) / torch.sum(torch.pow(real - est, 2), 1)), 0) / real.size()[0]
        return seg_sng

    @staticmethod
    def seg_SNR_(est, real):
        """
        预测的语音和真实语音的segSNR
        :param est:(B,T,F) 预测的语音
        :param real:(B,T,F) 真实的语音
        :return:分段SNR
        """
        # seg_snr = ((real.pow(2).sum(1) / (est - real).pow(2).sum(1)).log10() * 10).mean()
        seg_snr = (10.0 * torch.log10(torch.sum(torch.pow(real, 2), 1) / (torch.sum(torch.pow(real - est, 2), 1) + EPSILON))).mean()
        return seg_snr

    @staticmethod
    def mertics_LSD(est, label, is_full=True):
        """
        B,T,F
        :param est:
        :param label:
        :param is_full:
        :return:
        """
        if is_full:
            lsd = (label - est).pow(2).mean(2).sqrt().mean(1).mean(0)
            # lsd = (torch.log(label + EPSILON) - torch.log(est + EPSILON)).pow(2).mean(2).sqrt().mean(1).mean(0)
        else:
            est_half = est[:, :, 58:]
            label_half = label[:, :, 58:]
            lsd = (label_half - est_half).pow(2).mean(2).sqrt().mean(1).mean(0)
        return lsd
