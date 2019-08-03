import torch.nn.functional as F
import torch
import torch.nn as nn
from utils.util import pixel_shuffle_1d


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
        conv1_output = self.LeakyReLU1(self.bn1(self.conv1(x)))
        conv2_output = self.LeakyReLU2(self.bn2(self.conv2(conv1_output)))
        conv3_output = self.LeakyReLU3(self.bn3(self.conv3(conv2_output)))
        # (1,1024,2)
        conv_bottleneck_output = self.LeakyReLU_bottleneck(self.bn_bottleneck(self.conv_bottleneck(conv3_output)))
        conv5_output = self.LeakyReLU5(self.bn5(pixel_shuffle_1d(self.conv5(conv_bottleneck_output), 2)))

        # conv6
        stack1_output = torch.cat((conv5_output, conv3_output), 1)
        conv6_output = self.LeakyReLU6(self.bn6(pixel_shuffle_1d(self.conv6(stack1_output), 2)))
        # conv7
        stack2_output = torch.cat((conv6_output, conv2_output), 1)
        conv7_output = self.LeakyReLU7(self.bn7(pixel_shuffle_1d(self.conv7(stack2_output), 2)))
        # conv8
        stack3_output = torch.cat((conv7_output, conv1_output), 1)
        conv8_output = self.LeakyReLU8(self.bn8(pixel_shuffle_1d(self.conv8(stack3_output), 2)))
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
        self.conv1 = nn.Conv2d(in_channels=136, out_channels=512, kernel_size=(7, 1), stride=2, padding=(3, 0))
        # self.conv1 = nn.Conv1d(in_channels=81, out_channels=512, kernel_size=7, stride=2, padding=3)
        self.LeakyReLU1 = nn.LeakyReLU(negative_slope=0.2)
        # self.conv2 = nn.Conv1d(in_channels=512, out_channels=512, kernel_size=5, stride=2, padding=2)
        self.conv2 = nn.Conv2d(in_channels=512, out_channels=512, kernel_size=(5, 1), stride=2, padding=(2, 0))
        self.LeakyReLU2 = nn.LeakyReLU(negative_slope=0.2)
        # self.conv3 = nn.Conv1d(in_channels=512, out_channels=512, kernel_size=3, stride=2, padding=1)
        self.conv3 = nn.Conv2d(in_channels=512, out_channels=512, kernel_size=(3, 1), stride=2, padding=(1, 0))
        self.LeakyReLU3 = nn.LeakyReLU(negative_slope=0.2)
        # self.Linear1 = nn.Linear(in_features=2048, out_features=1024)
        self.Linear1 = nn.Linear(in_features=2048, out_features=1024)
        self.LeakyReLU4 = nn.LeakyReLU(negative_slope=0.2)
        # self.Linear2 = nn.Linear(in_features=1024, out_features=1)
        self.Linear2 = nn.Linear(in_features=1024, out_features=1)

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