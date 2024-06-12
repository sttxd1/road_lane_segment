import torch
from torch import nn
import torch.nn.functional as F


class ResBlock(nn.Module):
    def __init__(self, in_channels, mid_channels, out_channels, stride=1, first_dilation=None, dilation=1):
        super(ResBlock, self).__init__()

        self.same_shape = (in_channels == out_channels and stride == 1)

        if first_dilation is None:
            first_dilation = dilation

        self.bn_branch2a = nn.BatchNorm2d(in_channels)

        self.conv_branch2a = nn.Conv2d(in_channels, mid_channels, 3, stride,
                                       padding=first_dilation, dilation=first_dilation, bias=False)

        self.bn_branch2b1 = nn.BatchNorm2d(mid_channels)

        self.conv_branch2b1 = nn.Conv2d(mid_channels, out_channels, 3, padding=dilation, dilation=dilation, bias=False)

        if not self.same_shape:
            self.conv_branch1 = nn.Conv2d(in_channels, out_channels, 1, stride, bias=False)

    def forward(self, x, get_x_bn_relu=False):

        branch2 = self.bn_branch2a(x)
        branch2 = F.relu(branch2)

        x_bn_relu = branch2

        if not self.same_shape:
            branch1 = self.conv_branch1(branch2)
        else:
            branch1 = x

        branch2 = self.conv_branch2a(branch2)
        branch2 = self.bn_branch2b1(branch2)
        branch2 = F.relu(branch2)
        branch2 = self.conv_branch2b1(branch2)

        x = branch1 + branch2

        if get_x_bn_relu:
            return x, x_bn_relu

        return x

    def __call__(self, x, get_x_bn_relu=False):
        return self.forward(x, get_x_bn_relu=get_x_bn_relu)


class ResBlockBottle(nn.Module):
    def __init__(self, in_channels, out_channels, stride=1, dilation=1, dropout=0.):
        super(ResBlockBottle, self).__init__()

        self.same_shape = (in_channels == out_channels and stride == 1)

        self.bn_branch2a = nn.BatchNorm2d(in_channels)
        self.conv_branch2a = nn.Conv2d(in_channels, out_channels // 4, 1, stride, bias=False)

        self.bn_branch2b1 = nn.BatchNorm2d(out_channels // 4)
        self.dropout_2b1 = torch.nn.Dropout2d(dropout)
        self.conv_branch2b1 = nn.Conv2d(out_channels // 4, out_channels // 2, 3, padding=dilation, dilation=dilation, bias=False)

        self.bn_branch2b2 = nn.BatchNorm2d(out_channels // 2)
        self.dropout_2b2 = torch.nn.Dropout2d(dropout)
        self.conv_branch2b2 = nn.Conv2d(out_channels // 2, out_channels, 1, bias=False)

        if not self.same_shape:
            self.conv_branch1 = nn.Conv2d(in_channels, out_channels, 1, stride, bias=False)

    def forward(self, x, get_x_bn_relu=False):

        branch2 = self.bn_branch2a(x)
        branch2 = F.relu(branch2)
        x_bn_relu = branch2

        branch1 = self.conv_branch1(branch2)

        branch2 = self.conv_branch2a(branch2)

        branch2 = self.bn_branch2b1(branch2)
        branch2 = F.relu(branch2)
        branch2 = self.dropout_2b1(branch2)
        branch2 = self.conv_branch2b1(branch2)

        branch2 = self.bn_branch2b2(branch2)
        branch2 = F.relu(branch2)
        branch2 = self.dropout_2b2(branch2)
        branch2 = self.conv_branch2b2(branch2)

        x = branch1 + branch2

        if get_x_bn_relu:
            return x, x_bn_relu

        return x

    def __call__(self, x, get_x_bn_relu=False):
        return self.forward(x, get_x_bn_relu=get_x_bn_relu)


class ResNet38(nn.Module):
    def __init__(self, sequential_func=nn.Sequential):
        super(ResNet38, self).__init__()

        self.conv1 = nn.Conv2d(3, 64, 2, padding=1, bias=False)

        self.layer1 = sequential_func(
            ResBlock(64, 128, 128, stride=2),
            ResBlock(128, 128, 128),
            ResBlock(128, 128, 128)
        )

        self.layer2 = sequential_func(
            ResBlock(128, 256, 256, stride=2),
            ResBlock(256, 256, 256),
            ResBlock(256, 256, 256)
        )

        self.layer3 = sequential_func(
            ResBlock(256, 512, 512, stride=2),
            ResBlock(512, 512, 512),
            ResBlock(512, 512, 512),
            ResBlock(512, 512, 512),
            ResBlock(512, 512, 512),
            ResBlock(512, 512, 512),
        )

        self.layer4 = sequential_func(
            ResBlock(512, 512, 1024, stride=1, first_dilation=1, dilation=2),
            ResBlock(1024, 512, 1024, dilation=2),
            ResBlock(1024, 512, 1024, dilation=2),
            ResBlockBottle(1024, 2048, stride=1, dilation=4, dropout=0.3),
            ResBlockBottle(2048, 4096, dilation=4, dropout=0.5),
            nn.BatchNorm2d(4096),
        )
        
        self.sem_head = nn.Conv2d(4096, 2, kernel_size=3)
        self.ins_head = nn.Conv2d(4096, 5, kernel_size=3)

        self.not_training = [self.conv1]

    def forward(self, x):

        # print(f"Input shape: {x.shape}")

        x = self.conv1(x)
        # print(f"After conv1: {x.shape}")

        x = self.layer1(x)
        # print(f"After layer1: {x.shape}")

        x = self.layer2(x)
        # print(f"After layer2: {x.shape}")

        x = self.layer3(x)
        # print(f"After layer3: {x.shape}")

        x = self.layer4(x)
        # print(f"After layer4: {x.shape}")

        x = F.relu(x)
        # print(f"After ReLU: {x.shape}")

        sem_pred = self.sem_head(x)
        # print(f"sem_pred shape: {sem_pred.shape}")

        ins_pred = self.ins_head(x)
        # print(f"ins_pred shape: {ins_pred.shape}")
        
        ins_pred_resized = F.interpolate(ins_pred, size=(224, 224), mode='bicubic', align_corners=False)
        sem_pred_resized = F.interpolate(sem_pred, size=(224, 224), mode='bicubic', align_corners=False)
        # print(f"sem_pred shape: {ins_pred_resized.shape}")
        # print(f"ins_pred shape: {sem_pred_resized.shape}")

        return sem_pred_resized, ins_pred_resized
