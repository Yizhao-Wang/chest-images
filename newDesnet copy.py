

import os
import torch
from torch import nn
import math

from models import DenseNet, MiddleFeatures
import torch.nn.functional as F
from models import op_norm

def crop(data1, data2):
    _, _, h1, w1 = data1.size()
    _, _, h2, w2 = data2.size()
    if h1 < h2 or w1 < w2:
        pad_h = (h2 - h1) // 2 + 1
        pad_h = pad_h if pad_h > 0 else 0
        pad_w = (w2 - w1) // 2 + 1
        pad_w = pad_w if pad_w > 0 else 0
        data1 = torch.nn.ConstantPad2d((pad_w, pad_w, pad_h, pad_h), 0)(data1)
        _, _, h1, w1 = data1.size()
    assert (h2 <= h1 and w2 <= w1)
    offset_h = (h1 - h2) // 2
    offset_w = (w1 - w2) // 2
    data = data1[:, :, offset_h:offset_h + h2, offset_w:offset_w + w2]
    return data

def conv3x3(in_planes, out_planes, stride=1):
    """3x3 convolution with padding"""
    return nn.Conv2d(in_planes, out_planes, kernel_size=3, stride=stride,
                     padding=1, bias=True)


def conv1x1(in_planes, out_planes, stride=1):
    """1x1 convolution"""
    return nn.Conv2d(in_planes, out_planes, kernel_size=1, stride=stride, bias=True)


class ocpX2(nn.Module):  # self-attention module

    def __init__(self, planes, kernel_size1=1, kernel_size2=3):
        super(ocpX2, self).__init__()
        self.conv1 = torch.nn.Sequential(
            conv1x1(planes, int(planes / 2)),
            nn.BatchNorm2d(int(planes / 2)),
            nn.ReLU(inplace=True),
        )
        self.conv2 = torch.nn.Sequential(
            conv1x1(planes, int(planes / 2)),
            nn.BatchNorm2d(int(planes / 2)),
            nn.ReLU(inplace=True),
            nn.MaxPool2d(kernel_size=3, stride=2, padding=1)
        )
        # self.conv2 = self.conv1 = torch.nn.Sequential(
        self.conv3 = torch.nn.Sequential(
            conv1x1(int(planes/2), planes ),
            nn.BatchNorm2d(planes),
            nn.ReLU(inplace=True),
        )


    def forward(self, a):
        ori = a
        x1 = self.conv1(a)
        x1 = x1.reshape([x1.size()[0], -1, x1.size()[1]])
        x2 = self.conv2(a)
        x2 = x2.reshape([x2.size()[0], x2.size()[1], -1])
        sum1 = torch.matmul(x1, x2)
        net_1 = nn.Softmax(dim=2)
        sum1 = net_1(sum1)
        x3 = self.conv2(a)
        x3 = x3.reshape([x3.size()[0], -1, x3.size()[1]])
        sum2 = torch.matmul(sum1, x3)
        sum2 = sum2.reshape([a.size()[0], -1, a.size()[2], a.size()[3]])
        sum2 = self.conv3(sum2)
        sum2 = sum2.add(ori) 

        return sum2


class Dense_Nonlocal(torch.nn.Module):
    def __init__(self, weights=None, num_classes=1,in_channels=1, apply_sigmoid=False,args=None):
        # C64
        super(Dense_Nonlocal, self).__init__()
        self.args = args


        self.model = DenseNet(num_classes=num_classes, in_channels=in_channels)
        # self.pathologies = self.model.pathologies ####
        # self.middle = MiddleFeatures()
        # self.model.register_forward_hook(self.middle)
        in_c = [64, 256, 512, 1024, 2048]
        self.encoder_sides = nn.ModuleList()
        for i in range(len(self.model.block_config)+1):
            if i == 0:
                self.encoder_sides.append(torch.nn.Sequential(
                    ocpX2(in_c[i]),
                    conv1x1(in_c[i], int(int(in_c[i])*2)),
                    nn.BatchNorm2d(int(int(in_c[i])*2)),
                    nn.ReLU(inplace=True)
                    # nn.AvgPool2d(kernel_size=2, stride=2)
                ))
            elif i == (len(self.model.block_config)):
                self.encoder_sides.append(torch.nn.Sequential(
                    ocpX2(in_c[i]),
                    # conv1x1(in_c[i], in_c[i-1]),
                    nn.BatchNorm2d(in_c[i]),
                    nn.ReLU(inplace=True)
    
                ))
            else:
                self.encoder_sides.append(torch.nn.Sequential(
                    ocpX2(in_c[i]),
                    nn.BatchNorm2d(in_c[i]),
                    nn.ReLU(inplace=True)
                    # nn.AvgPool2d(kernel_size=2, stride=2)
                ))
        # num_classes = len(self.pathologies)####
        self.classifier = nn.Linear(in_c[len(self.model.block_config)], num_classes)#
        # self.classifier = nn.Linear(128, num_classes)#
        self._initialize_weights()

    def forward(self, inputs):
        middle_features = self.model(inputs)
        # nonlocal_feature = middle_features[0]
        nonlocal_feature = self.encoder_sides[0](middle_features[0])
        # print("nonlocal_feature=",nonlocal_feature.size())
        # print(self.encoder_sides)
        for i in range(len(middle_features)):
            
            if i != 0:
                nonlocal_feature = crop(nonlocal_feature, middle_features[i])
                merge_feature = torch.cat([nonlocal_feature, middle_features[i]], 1)
                # print("nonlocal_feature=",merge_feature.size())
                # print("middl=",middle_features[i].size())
                nonlocal_feature = self.encoder_sides[i](merge_feature)


        # nonlocal_feature = crop(nonlocal_feature, middle_features[1])
        # nonlocal_feature = torch.cat([nonlocal_feature, middle_features[1]], 1)
        # print("nonlocal_feature=",nonlocal_feature.size())
        # print("middl=",middle_features[1].size())
        # nonlocal_feature = self.encoder_sides[1](nonlocal_feature)


        # out = F.relu(nonlocal_feature, inplace=True)
        out = F.adaptive_avg_pool2d(nonlocal_feature, (1, 1)).view(nonlocal_feature.size(0), -1)
        out = self.classifier(out)
        if hasattr(self.model, 'apply_sigmoid') and self.model.apply_sigmoid:
            out = torch.sigmoid(out)

        if hasattr(self.model, "op_threshs") and (self.model.op_threshs != None):
            out = torch.sigmoid(out)
            out = op_norm(out, self.model.op_threshs)


        return out

    def _initialize_weights(self):

        for m in self.modules():
            if isinstance(m, nn.Conv2d):                  
                nn.init.kaiming_uniform_(m.weight, a=math.sqrt(5))
                if m.bias is not None:
                    nn.init.constant_(m.bias, 0)
            elif isinstance(m, (nn.BatchNorm2d, nn.GroupNorm)):
                nn.init.constant_(m.bias, 0)
            elif isinstance(m, nn.ConvTranspose2d):
                m.weight.data = module.bilinear_kernel(m.in_channels, m.out_channels, m.kernel_size[0])



