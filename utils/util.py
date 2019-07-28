import os, struct
import time
import re
import torch
from config import *
import scipy.signal as signal
import scipy.io.wavfile as wavfile
from utils.stft_istft import STFT
import numpy as np
from torch.autograd import Variable
import tensorflow as tf


"""sgementaxis code.

This code has been implemented by Anne Archibald, and has been discussed on the
ML."""

import numpy as np
import warnings

def segment_axis(a, length, overlap=0, axis=None, end='cut', endvalue=0):
    """Generate a new array that chops the given array along the given axis
    into overlapping frames.

    example:
    >>> segment_axis(arange(10), 4, 2)
    array([[0, 1, 2, 3],
           [2, 3, 4, 5],
           [4, 5, 6, 7],
           [6, 7, 8, 9]])

    arguments:
    a       The array to segment
    length  The length of each frame
    overlap The number of array elements by which the frames should overlap
    axis    The axis to operate on; if None, act on the flattened array
    end     What to do with the last frame, if the array is not evenly
            divisible into pieces. Options are:

            'cut'   Simply discard the extra values
            'wrap'  Copy values from the beginning of the array
            'pad'   Pad with a constant value

    endvalue    The value to use for end='pad'

    The array is not copied unless necessary (either because it is unevenly
    strided and being flattened or because end is set to 'pad' or 'wrap').
    """

    if axis is None:
        a = np.ravel(a) # may copy
        axis = 0

    l = a.shape[axis]

    if overlap >= length:
        print("frames cannot overlap by more than 100%")
    if overlap < 0 or length <= 0:
        print("overlap must be nonnegative and length must "\
                          "be positive")

    if l < length or (l-length) % (length-overlap):
        if l>length:
            roundup = length + (1+(l-length)//(length-overlap))*(length-overlap)
            rounddown = length + ((l-length)//(length-overlap))*(length-overlap)
        else:
            roundup = length
            rounddown = 0
        assert rounddown < l < roundup
        assert roundup == rounddown + (length-overlap) \
               or (roundup == length and rounddown == 0)
        a = a.swapaxes(-1,axis)

        if end == 'cut':
            a = a[..., :rounddown]
        elif end in ['pad','wrap']: # copying will be necessary
            s = list(a.shape)
            s[-1] = roundup
            b = np.empty(s,dtype=a.dtype)
            b[..., :l] = a
            if end == 'pad':
                b[..., l:] = endvalue
            elif end == 'wrap':
                b[..., l:] = a[..., :roundup-l]
            a = b

        a = a.swapaxes(-1,axis)


    l = a.shape[axis]
    if l == 0:
          print("Not enough data points to segment array in 'cut' mode; "\
              "try 'pad' or 'wrap'")
    assert l >= length
    assert (l-length) % (length-overlap) == 0
    n = 1 + (l-length) // (length-overlap)
    s = a.strides[axis]
    newshape = a.shape[:axis] + (n,length) + a.shape[axis+1:]
    newstrides = a.strides[:axis] + ((length-overlap)*s,s) + a.strides[axis+1:]

    try:
        return np.ndarray.__new__(np.ndarray, strides=newstrides,
                                  shape=newshape, buffer=a, dtype=a.dtype)
    except TypeError:
        warnings.warn("Problem with ndarray creation forces copy.")
        a = a.copy()
        # Shape doesn't change but strides does
        newstrides = a.strides[:axis] + ((length-overlap)*s,s) \
                     + a.strides[axis+1:]
        return np.ndarray.__new__(np.ndarray, strides=newstrides,
                                  shape=newshape, buffer=a, dtype=a.dtype)



def expandWindow(data, left, right):
    data = data.detach().cpu().numpy()
    sp = data.shape
    idx = 0
    exdata = np.zeros([sp[0], sp[1], sp[2] * (left + right + 1)])
    for i in range(-left, right+1):
        exdata[:, :, sp[2] * idx : sp[2] * (idx + 1)] = np.roll(data, shift=-i, axis=1)
        idx = idx + 1
    return torch.Tensor(exdata).cuda(CUDA_ID[0])


def context_window(data, left, right):
    """
    扩帧函数
    :param data:tensor类型的待扩帧数据，shape=(B,T,F)
    :param left: 左扩长度
    :param right: 右扩长度
    :return: 扩帧后的结果 shape = (B,T,F * (1 + left + right))
    """
    sp = data.size()
    # exdata = torch.zeros(sp[0], sp[1], sp[2] * (left + right + 1)).cuda(CUDA_ID[0])
    exdata = torch.zeros(sp[0], sp[1], sp[2] * (left + right + 1)).cuda(CUDA_ID[0])
    for i in range(1, left + 1):
        exdata[:, i:, sp[2] * (left - i) : sp[2] * (left - i + 1)] = data[:, :-i,:]
    for i in range(1, right+1):
        exdata[:, :-i, sp[2] * (left + i):sp[2]*(left+i+1)] = data[:, i:, :]
    exdata[:, :, sp[2] * left : sp[2] * (left + 1)] = data
    return exdata


def read_list(list_file):
    f = open(list_file, "r")
    lines = f.readlines()
    list_sig = []
    for x in lines:
        list_sig.append(x[:-1])
    f.close()
    return list_sig


def gen_list(wav_dir, append):
    """使用正则表达式获取相应文件的list
    wav_dir:路径
    append:文件类型，eg: .wav .mat
    """
    l = []
    lst = os.listdir(wav_dir)
    lst.sort()
    for f in lst:
        if re.search(append, f):
            l.append(f)
    return l


def write_log(file, name, train, validate):
    message = ''
    for m, val in enumerate(train):
        message += ' --TRerror%i=%.3f ' % (m, val.data.numpy())
    for m, val in enumerate(validate):
        message += ' --CVerror%i=%.3f ' % (m, val.data.numpy())
    file.write(name + ' ')
    file.write(message)
    file.write('/n')


def get_alpha(mix, constant=1):
    """
    求得进行能量归一化的alpha值
    :param mix: 带噪语音的采样点的array
    :param constant: 一般取值为1，即使噪声平均每个采样点的能量在1以内
    :return: 权值c
    """
    # c = np.sqrt(constant * mix.size / np.sum(mix**2)), s *= c, mix *= c
    return np.sqrt(constant * mix.size / np.sum(mix ** 2))


def wav_file_resample(src, dst, source_sample=44100, dest_sample=16000):
    """
    对WAV文件进行resample的操作
    :param file_path: 需要进行resample操作的wav文件的路径
    :param source_sample:原始采样率
    :param dest_sample:目标采样率
    :return:
    """
    sample_rate, sig = wavfile.read(src)
    result = int((sig.shape[0]) / source_sample * dest_sample)
    x_resampled = signal.resample(sig, result)
    x_resampled = x_resampled.astype(np.float64)
    return x_resampled, dest_sample
    # wavfile.write(dst, dest_sample, x_resampled)


def pixel_shuffle_1d(inp, upscale_factor=2):
    batch_size, channels, in_width = inp.size()
    channels //= upscale_factor

    out_width = in_width * upscale_factor
    inp_view = inp.contiguous().view(batch_size, channels, upscale_factor, in_width)
    shuffle_out = inp_view.permute(0, 1, 3, 2).contiguous()
    shuffle_out = shuffle_out.view(batch_size, channels, out_width)
    return shuffle_out


def write_bin_file(source_dir, dest_file):
    work_dir = source_dir
    with open(dest_file, 'wb') as file:
        for parent, dirnames, filenames in os.walk(work_dir):
            for filename in filenames:
                file_path = os.path.join(parent, filename)
                if filename.lower().endswith('.wav'):
                    try:
                        start = time.time() * 1000
                        print('读取到WAV文件: {}'.format(file_path))
                        sample, signal = wavfile.read(file_path)
                        assert signal.dtype == np.int16
                        signal = signal / 32768.0
                        for i in signal:
                            f = struct.pack('f', i)
                            file.write(f)
                        end = time.time() * 1000
                        print("花费时间：{}".format(str(end - start)))
                    except:
                        pass


def cal_mean_var(speech, is_log=IS_LOG):
    stft = STFT(filter_length=FILTER_LENGTH, hop_length=HOP_LENGTH)
    speech_spec = stft.transform(speech)
    speech_real = speech_spec[:, :, :, 0]
    speech_imag = speech_spec[:, :, :, 1]
    speech_mag = torch.sqrt(speech_real ** 2 + speech_imag ** 2).squeeze()
    if is_log:
        speech_mag = torch.log(speech_mag)
    mean = torch.mean(speech_mag, 0)
    var = torch.var(speech_mag, 0)
    return mean, var


def normalization(abs_data):
    """
    归一化
    :param input_data:tensor,abs谱
    :return: 归一化后的谱
    """
    abs_data[abs_data < 1e-4] = 1e-4
    # s = abs_data.clamp(1e-4, 10000)
    s = 20 * torch.log(abs_data + EPSILON) - 20
    s = (s + 100) / 100
    s = s.clamp(0, 1)
    return s


def re_normalization(normalization_data):
    """
    恢复normailzation的数据
    :param normalization_data:归一化的数据
    :return:愿数据，abs谱
    """
    s = normalization_data * 100 - 100
    s = torch.exp((s + 20) / 20) - EPSILON
    s[s <= (np.exp(-4) + 1e-4)] = 0
    return s


def discriminator_regularizer(D1_logits, D1_arg, D2_logits, D2_arg, net, optim):
    D2 = torch.sigmoid(D2_logits)
    D1 = torch.sigmoid(D1_logits)
    D1_logits.mean().backward(retain_graph=True)
    grad_D1_logits = D1_arg.grad
    net.zero_grad()
    optim.zero_grad()
    D2_logits.mean().backward(retain_graph=True)
    grad_D2_logits = D2_arg.grad
    net.zero_grad()
    optim.zero_grad()
    grad_D1_logits_norm = torch.norm(grad_D1_logits.reshape(TRAIN_BATCH_SIZE, -1), dim=1)
    grad_D2_logits_norm = torch.norm(grad_D2_logits.reshape(TRAIN_BATCH_SIZE, -1), dim=1)
    reg_D1 = torch.pow(1.0-D1, 2) * torch.pow(grad_D1_logits_norm, 2)
    reg_D2 = torch.pow(D2, 2) * torch.pow(grad_D2_logits_norm, 2)
    disc_regularizer = torch.mean(reg_D1 + reg_D2)
    return disc_regularizer


def vec2frame(signal, frame_len, overlap):
    y = segment_axis(signal, frame_len, overlap=overlap, end='cut')
    return y