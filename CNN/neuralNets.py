###################################################################################################
# Imports
###################################################################################################
import numpy as np
import math

import torch
import torch.nn as nn
import torch.nn.functional as F
import torchvision

###################################################################################################
# Classes and Functions
###################################################################################################


class Net(nn.Module):
    def __init__(self):
        super(Net, self).__init__()
        self.conv1 = nn.Conv2d(
            2, 6, 3, padding=1
        )  # input image channel, output channels, 3x3 square convolution kernel, padding
        self.conv2 = nn.Conv2d(6, 16, 3, padding=1)
        self.conv3 = nn.Conv2d(16, 32, 3, padding=1)
        self.conv4 = nn.Conv2d(32, 64, 3, padding=1)
        self.conv5 = nn.Conv2d(64, 128, 3, padding=1)
        self.fc1 = nn.Linear(128 * 8 * 8, 1000)
        self.fc2 = nn.Linear(1000, 100)
        self.fc3 = nn.Linear(100, 6)
        self.dropout = nn.Dropout(0.25)
        # self.adPool = nn.AdaptiveMaxPool2d(8)

    def forward(self, x):
        x = F.max_pool2d(F.relu(self.conv1(x)), 2)
        x = F.max_pool2d(F.relu(self.conv2(x)), 2)
        x = F.max_pool2d(F.relu(self.conv3(x)), 2)
        x = F.max_pool2d(F.relu(self.conv4(x)), 2)
        x = F.max_pool2d(F.relu(self.conv5(x)), 2)
        x = x.view(-1, self.num_flat_features(x))
        # x = self.dropout(x)
        x = F.relu(self.fc1(x))
        # x = self.dropout(x)
        x = F.relu(self.fc2(x))
        # x = self.dropout(x)
        x = self.fc3(x)
        return x

    def num_flat_features(self, x):
        size = x.size()[1:]
        num_features = 1
        for s in size:
            num_features *= s
        return num_features


class Net_bn(nn.Module):
    def __init__(self):
        super(Net_bn, self).__init__()
        self.conv1 = nn.Conv2d(
            2, 6, 3, padding=1, bias=False
        )  # input image channel, output channels, 3x3 square convolution kernel, padding
        self.bn1 = nn.BatchNorm2d(6)
        self.conv2 = nn.Conv2d(6, 16, 3, padding=1, bias=False)
        self.bn2 = nn.BatchNorm2d(16)
        self.conv3 = nn.Conv2d(16, 32, 3, padding=1, bias=False)
        self.bn3 = nn.BatchNorm2d(32)
        self.conv4 = nn.Conv2d(32, 64, 3, padding=1, bias=False)
        self.bn4 = nn.BatchNorm2d(64)
        self.conv5 = nn.Conv2d(64, 128, 3, padding=1, bias=False)
        self.bn5 = nn.BatchNorm2d(128)
        self.fc1 = nn.Linear(128 * 8 * 8, 1000)
        self.fc2 = nn.Linear(1000, 100)
        self.fc3 = nn.Linear(100, 6)
        self.dropout = nn.Dropout(0.5)
        # self.adPool = nn.AdaptiveMaxPool2d(8)

    def forward(self, x):
        x = F.max_pool2d(F.relu(self.bn1(self.conv1(x))), 2)
        x = F.max_pool2d(F.relu(self.bn2(self.conv2(x))), 2)
        x = F.max_pool2d(F.relu(self.bn3(self.conv3(x))), 2)
        x = F.max_pool2d(F.relu(self.bn4(self.conv4(x))), 2)
        x = F.max_pool2d(F.relu(self.bn5(self.conv5(x))), 2)
        x = x.view(-1, self.num_flat_features(x))
        x = self.dropout(x)
        x = F.relu(self.fc1(x))
        x = self.dropout(x)
        x = F.relu(self.fc2(x))
        x = self.dropout(x)
        x = self.fc3(x)
        return x

    def num_flat_features(self, x):
        size = x.size()[1:]
        num_features = 1
        for s in size:
            num_features *= s
        return num_features


class Net_bn0(nn.Module):
    def __init__(self):
        super(Net_bn0, self).__init__()
        self.bn0 = nn.BatchNorm2d(2)
        self.conv1 = nn.Conv2d(
            2, 6, 3, padding=1, bias=False
        )  # input image channel, output channels, 3x3 square convolution kernel, padding
        self.bn1 = nn.BatchNorm2d(6)
        self.conv2 = nn.Conv2d(6, 16, 3, padding=1, bias=False)
        self.bn2 = nn.BatchNorm2d(16)
        self.conv3 = nn.Conv2d(16, 32, 3, padding=1, bias=False)
        self.bn3 = nn.BatchNorm2d(32)
        self.conv4 = nn.Conv2d(32, 64, 3, padding=1, bias=False)
        self.bn4 = nn.BatchNorm2d(64)
        self.conv5 = nn.Conv2d(64, 128, 3, padding=1, bias=False)
        self.bn5 = nn.BatchNorm2d(128)
        self.fc1 = nn.Linear(128 * 8 * 8, 1000)
        self.fc2 = nn.Linear(1000, 100)
        self.fc3 = nn.Linear(100, 6)
        self.dropout = nn.Dropout(0.5)
        # self.adPool = nn.AdaptiveMaxPool2d(8)

    def forward(self, x):
        x = self.bn0(x)
        x = F.max_pool2d(F.relu(self.bn1(self.conv1(x))), 2)
        x = F.max_pool2d(F.relu(self.bn2(self.conv2(x))), 2)
        x = F.max_pool2d(F.relu(self.bn3(self.conv3(x))), 2)
        x = F.max_pool2d(F.relu(self.bn4(self.conv4(x))), 2)
        x = F.max_pool2d(F.relu(self.bn5(self.conv5(x))), 2)
        x = x.view(-1, self.num_flat_features(x))
        x = self.dropout(x)
        x = F.relu(self.fc1(x))
        x = self.dropout(x)
        x = F.relu(self.fc2(x))
        x = self.dropout(x)
        x = self.fc3(x)
        return x

    def num_flat_features(self, x):
        size = x.size()[1:]
        num_features = 1
        for s in size:
            num_features *= s
        return num_features


class Net_bn_short(nn.Module):
    def __init__(self, p_dropout):
        super(Net_bn_short, self).__init__()
        self.bn0 = nn.BatchNorm2d(2)
        self.conv1 = nn.Conv2d(
            2, 6, 3, padding=1, bias=False
        )  # input image channel, output channels, 3x3 square convolution kernel, padding
        self.bn1 = nn.BatchNorm2d(6)
        self.conv2 = nn.Conv2d(6, 16, 3, padding=1, bias=False)
        self.bn2 = nn.BatchNorm2d(16)
        self.conv3 = nn.Conv2d(16, 32, 3, padding=1, bias=False)
        self.bn3 = nn.BatchNorm2d(32)
        self.conv4 = nn.Conv2d(32, 64, 3, padding=1, bias=False)
        self.bn4 = nn.BatchNorm2d(64)
        self.conv5 = nn.Conv2d(64, 128, 3, padding=1, bias=False)
        self.bn5 = nn.BatchNorm2d(128)
        self.fc1 = nn.Linear(128 * 8 * 8, 1000)
        self.fc2 = nn.Linear(1000, 6)
        self.dropout = nn.Dropout(p_dropout)

    def forward(self, x):
        x = F.relu(self.bn0(x))
        x = F.max_pool2d(F.relu(self.bn1(self.conv1(x))), 2)
        x = F.max_pool2d(F.relu(self.bn2(self.conv2(x))), 2)
        x = F.max_pool2d(F.relu(self.bn3(self.conv3(x))), 2)
        x = F.max_pool2d(F.relu(self.bn4(self.conv4(x))), 2)
        x = F.max_pool2d(F.relu(self.bn5(self.conv5(x))), 2)
        x = x.view(-1, self.num_flat_features(x))
        x = self.dropout(x)
        x = F.relu(self.fc1(x))
        x = self.dropout(x)
        x = self.fc2(x)
        return x

    def num_flat_features(self, x):
        size = x.size()[1:]
        num_features = 1
        for s in size:
            num_features *= s
        return num_features


class Net_ultaShort(nn.Module):
    def __init__(self):
        super(Net_ultaShort, self).__init__()
        self.bn0 = nn.BatchNorm2d(2)
        self.conv1 = nn.Conv2d(
            2, 4, 3, stride=1, padding=1, bias=False, groups=2
        )  # input image channel, output channels, 3x3 square convolution kernel, padding
        self.bn1 = nn.BatchNorm2d(4)
        self.conv2 = nn.Conv2d(4, 8, 3, stride=1, padding=1, bias=False, groups=2)
        self.bn2 = nn.BatchNorm2d(8)
        self.conv3 = nn.Conv2d(8, 16, 3, stride=1, padding=1, bias=False, groups=2)
        self.bn3 = nn.BatchNorm2d(16)

        self.fc1 = nn.Linear(16 * 8 * 8, 16 * 8 * 8)
        self.fc2 = nn.Linear(16 * 8 * 8, 6)
        self.dropout = nn.Dropout(0.5)

    def forward(self, x):
        x = F.relu(self.bn0(x))
        x = F.max_pool2d(F.relu(self.bn1(self.conv1(x))), 4)
        x = F.max_pool2d(F.relu(self.bn2(self.conv2(x))), 4)
        x = F.max_pool2d(F.relu(self.bn3(self.conv3(x))), 2)

        x = x.view(-1, self.num_flat_features(x))
        x = self.dropout(x)
        x = F.relu(self.fc1(x))
        x = self.dropout(x)
        x = self.fc2(x)
        return x

    def num_flat_features(self, x):
        size = x.size()[1:]
        num_features = 1
        for s in size:
            num_features *= s
        return num_features


class VGG_custom(nn.Module):
    def __init__(self):
        super(VGG_custom, self).__init__()
        self.conv64_1 = nn.Conv2d(2, 64, 3, padding=1)
        self.conv64_2 = nn.Conv2d(64, 64, 3, padding=1)
        self.conv128_1 = nn.Conv2d(64, 128, 3, padding=1)
        self.conv128_2 = nn.Conv2d(128, 128, 3, padding=1)
        self.conv256_1 = nn.Conv2d(128, 256, 3, padding=1)
        self.conv256_2 = nn.Conv2d(256, 256, 3, padding=1)
        self.conv512_1 = nn.Conv2d(256, 512, 3, padding=1)
        self.conv512_2 = nn.Conv2d(512, 512, 3, padding=1)

        self.fc1 = nn.Linear(512 * 8 * 8, 4096)
        self.fc2 = nn.Linear(4096, 4096)
        self.fc3 = nn.Linear(4096, 6)
        self.dropout = nn.Dropout(0.5)
        # self.adPool = nn.AdaptiveMaxPool2d(8)

    def forward(self, x):
        x = F.relu(self.conv64_1(x))
        x = F.max_pool2d(F.relu(self.conv64_2(x)), 2)
        x = F.relu(self.conv128_1(x))
        x = F.max_pool2d(F.relu(self.conv128_2(x)), 2)
        x = F.relu(self.conv256_1(x))
        x = F.relu(self.conv256_2(x))
        x = F.relu(self.conv256_2(x))
        x = F.max_pool2d(F.relu(self.conv256_2(x)), 2)
        x = F.relu(self.conv512_1(x))
        x = F.relu(self.conv512_2(x))
        x = F.relu(self.conv512_2(x))
        x = F.max_pool2d(F.relu(self.conv512_2(x)), 2)
        x = F.relu(self.conv512_2(x))
        x = F.relu(self.conv512_2(x))
        x = F.relu(self.conv512_2(x))
        x = F.max_pool2d(F.relu(self.conv512_2(x)), 2)

        x = x.view(-1, self.num_flat_features(x))
        x = self.dropout(x)
        x = F.relu(self.fc1(x))
        x = self.dropout(x)
        x = F.relu(self.fc2(x))
        x = self.fc3(x)
        return x

    def num_flat_features(self, x):
        size = x.size()[1:]
        num_features = 1
        for s in size:
            num_features *= s
        return num_features
