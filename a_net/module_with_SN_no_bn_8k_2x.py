import torch.nn.functional as F
import torch
import torch.nn as nn
from torch.nn.modules.utils import _pair
from torch.nn.modules import conv, Linear
from utils.util import pixel_shuffle_1d


#define _l2normalization
def _l2normalize(v, eps=1e-12):
    return v / (torch.norm(v) + eps)


def max_singular_value(W, u=None, Ip=1):
    """
    power iteration for weight parameter
    """
    #xp = W.data
    if not Ip >= 1:
        raise ValueError("Power iteration should be a positive integer")
    if u is None:
        u = torch.FloatTensor(1, W.size(0)).normal_(0, 1).cuda()
    _u = u
    for _ in range(Ip):
        _v = _l2normalize(torch.matmul(_u, W.data), eps=1e-12)
        _u = _l2normalize(torch.matmul(_v, torch.transpose(W.data, 0, 1)), eps=1e-12)
    sigma = torch.sum(F.linear(_u, torch.transpose(W.data, 0, 1)) * _v)
    return sigma, _u


# noinspection PyTypeChecker
class GeneratorNet(nn.Module):
    def __init__(self):
        super(GeneratorNet, self).__init__()
        "encoder layers"
        # conv1
        self.conv1 = nn.Conv1d(in_channels=65, out_channels=256, kernel_size=7, stride=2, padding=3)
        self.bn1 = nn.BatchNorm1d(num_features=256)
        self.LeakyReLU1 = nn.LeakyReLU(negative_slope=0.2)
        # conv2
        self.conv2 = nn.Conv1d(in_channels=256, out_channels=512, kernel_size=5, stride=2, padding=2)
        self.bn2 = nn.BatchNorm1d(num_features=512)
        self.LeakyReLU2 = nn.LeakyReLU(negative_slope=0.2)
        # conv3
        self.conv3 = nn.Conv1d(in_channels=512, out_channels=512, kernel_size=3, stride=2, padding=1)
        self.bn3 = nn.BatchNorm1d(num_features=512)
        self.LeakyReLU3 = nn.LeakyReLU(negative_slope=0.2)
        "bottleneck layer"
        self.conv_bottleneck = nn.Conv1d(in_channels=512, out_channels=1024, kernel_size=3, stride=2, padding=1)
        self.bn_bottleneck = nn.BatchNorm1d(num_features=1024)
        self.LeakyReLU_bottleneck = nn.LeakyReLU(negative_slope=0.2)
        "decoder layers"
        # conv5
        self.conv5 = nn.Conv1d(in_channels=1024, out_channels=1024, kernel_size=3, stride=1, padding=1)
        self.bn5 = nn.BatchNorm1d(num_features=512)
        self.LeakyReLU5 = nn.LeakyReLU(negative_slope=0.2)
        # conv6
        self.conv6 = nn.Conv1d(in_channels=1024, out_channels=1024, kernel_size=5, stride=1, padding=2)
        self.bn6 = nn.BatchNorm1d(num_features=512)
        self.LeakyReLU6 = nn.LeakyReLU(negative_slope=0.2)
        # conv7
        self.conv7 = nn.Conv1d(in_channels=1024, out_channels=512, kernel_size=7, stride=1, padding=3)
        self.bn7 = nn.BatchNorm1d(num_features=256)
        self.LeakyReLU7 = nn.LeakyReLU(negative_slope=0.2)
        "upsample layer"
        self.conv8 = nn.Conv1d(in_channels=512, out_channels=71 * 2, kernel_size=7, stride=1, padding=3)
        self.bn8 = nn.BatchNorm1d(num_features=71)
        self.LeakyReLU8 = nn.LeakyReLU(negative_slope=0.2)
        "output layer"
        self.conv9 = nn.Conv1d(in_channels=71, out_channels=71, kernel_size=9, stride=1, padding=4)

    def forward(self, x):
        conv1_output = self.LeakyReLU1(self.conv1(x))
        conv2_output = self.LeakyReLU2(self.conv2(conv1_output))
        conv3_output = self.LeakyReLU3(self.conv3(conv2_output))
        # (1,1024,2)
        conv_bottleneck_output = self.LeakyReLU_bottleneck(self.bn_bottleneck(self.conv_bottleneck(conv3_output)))
        conv5_output = self.LeakyReLU5(pixel_shuffle_1d(self.conv5(conv_bottleneck_output), 2))

        # conv6
        stack1_output = torch.cat((conv5_output, conv3_output), 1)
        conv6_output = self.LeakyReLU6(pixel_shuffle_1d(self.conv6(stack1_output), 2))
        # conv7
        stack2_output = torch.cat((conv6_output, conv2_output), 1)
        conv7_output = self.LeakyReLU7(pixel_shuffle_1d(self.conv7(stack2_output), 2))
        # conv8
        stack3_output = torch.cat((conv7_output, conv1_output), 1)
        conv8_output = self.LeakyReLU8(pixel_shuffle_1d(self.conv8(stack3_output), 2))
        # conv9
        conv9_output = self.conv9(conv8_output)

        # return conv9_output.permute(0, 2, 1)
        # return F.softplus(conv9_output.permute(0, 2, 1))
        # return conv9_output.permute(0, 2, 1)
        # return F.relu(conv9_output.permute(0, 2, 1))
        return conv9_output.permute(0, 2, 1)


class DiscriminatorNet(nn.Module):
    def __init__(self):
        super(DiscriminatorNet, self).__init__()
        self.conv1 = SNConv2d(in_channels=136, out_channels=512, kernel_size=(7, 1), stride=2, padding=(3, 0))
        # self.conv1 = nn.Conv1d(in_channels=81, out_channels=512, kernel_size=7, stride=2, padding=3)
        self.LeakyReLU1 = nn.LeakyReLU(negative_slope=0.2)
        # self.conv2 = nn.Conv1d(in_channels=512, out_channels=512, kernel_size=5, stride=2, padding=2)
        self.conv2 = SNConv2d(in_channels=512, out_channels=512, kernel_size=(5, 1), stride=2, padding=(2, 0))
        self.LeakyReLU2 = nn.LeakyReLU(negative_slope=0.2)
        # self.conv3 = nn.Conv1d(in_channels=512, out_channels=512, kernel_size=3, stride=2, padding=1)
        self.conv3 = SNConv2d(in_channels=512, out_channels=512, kernel_size=(3, 1), stride=2, padding=(1, 0))
        self.LeakyReLU3 = nn.LeakyReLU(negative_slope=0.2)
        # self.Linear1 = nn.Linear(in_features=2048, out_features=1024)
        self.Linear1 = SNLinear(in_features=2048, out_features=1024)
        self.LeakyReLU4 = nn.LeakyReLU(negative_slope=0.2)
        # self.Linear2 = nn.Linear(in_features=1024, out_features=1)
        self.Linear2 = SNLinear(in_features=1024, out_features=1)

    def forward(self, x):
        conv1_output = self.LeakyReLU1(self.conv1(x))
        conv2_output = self.LeakyReLU2(self.conv2(conv1_output))
        conv3_output = self.LeakyReLU3(self.conv3(conv2_output))
        flatten_output = conv3_output.reshape(conv3_output.size()[0], -1)
        linear1_output = self.LeakyReLU4(self.Linear1(flatten_output))
        linear2_output = self.Linear2(linear1_output)
        output = linear2_output
        # output has no activation function
        # because the loss function is BCEWithLogitsLoss that combine sigmoid and BCELoss together
        return output


class SNLinear(Linear):
    def __init__(self, in_features, out_features, bias=True):
        super(SNLinear, self).__init__(in_features, out_features, bias)
        self.register_buffer('u', torch.Tensor(1, out_features).normal_())

    @property
    def W_(self):
        w_mat = self.weight.view(self.weight.size(0), -1)
        sigma, _u = max_singular_value(w_mat, self.u)
        self.u.copy_(_u)
        return self.weight / sigma

    def forward(self, input):
        return F.linear(input, self.W_, self.bias)


class SNConv2d(conv._ConvNd):
    def __init__(self, in_channels, out_channels, kernel_size, stride=1, padding=0, dilation=1, groups=1, bias=True):
        kernel_size = _pair(kernel_size)
        stride = _pair(stride)
        padding = _pair(padding)
        dilation = _pair(dilation)
        super(SNConv2d, self).__init__(
            in_channels, out_channels, kernel_size, stride, padding, dilation,
            False, _pair(0), groups, bias)
        self.register_buffer('u', torch.Tensor(1, out_channels).normal_())

    @property
    def W_(self):
        w_mat = self.weight.view(self.weight.size(0), -1)
        sigma, _u = max_singular_value(w_mat, self.u)
        self.u.copy_(_u)
        return self.weight / sigma

    def forward(self, input):
        return F.conv1d(input, self.W_, self.bias, self.stride,
                        self.padding, self.dilation, self.groups)
