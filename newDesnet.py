from __future__ import absolute_import, division, print_function

import os
import torch
from torch import nn
import math

from encoding.models import resnet_encoding as res
# from encoding.models import module
# import common.resnet_encoding as res
import common.module as module
from encoding.models import pathladderN


class Net(torch.nn.Module):
    def __init__(self, pretrain='../data/model/resnet50-19c8e357.pth', args=None):
        # C64
        super(Net, self).__init__()
        self.args = args
        self.res50_model_path = 'data/resnet50-25c4b509_encoding.pth'
        self.res50 = res.resnet50(dilated=True)


        self.model = Desnet()
        self.middle = MiddleFeatures()
        self.model.register_forward_hook(middle)


        self.pathladder = pathladderN.PathLadder(planes=[32, 128, 256, 512, 1024, 2048], x=8)

        self.convolutionX3 = torch.nn.Sequential(
            nn.Conv2d(3, 16, 3, padding=1, bias=True),
            nn.BatchNorm2d(16),
            nn.ReLU(inplace=True),
            nn.Conv2d(16, 32, 3, padding=1, bias=True),
            nn.BatchNorm2d(32),
            nn.ReLU(inplace=True)
        )
        self.edge_outsides = nn.ModuleList()
        for i in range(6):
            self.edge_outsides.append(torch.nn.Sequential(
                conv1x1(32, 1)
            ))
        self.fuse = torch.nn.Sequential(
            conv1x1(6, 1)
        )

        self.ori_convolutionX3 = torch.nn.Sequential(
            nn.Conv2d(3, 16, 3, padding=1, bias=True),
            nn.BatchNorm2d(16),
            nn.ReLU(inplace=True),
            nn.Conv2d(16, 32, 3, padding=1, bias=True),
            nn.BatchNorm2d(32),
            nn.ReLU(inplace=True),
            nn.Conv2d(32, 64, 3, padding=1, bias=True),
            nn.BatchNorm2d(64),
            nn.ReLU(inplace=True)
        )
        
        in_c = [64, 128, 256, 512, 1024, 2048]
        self.encoder_sides = nn.ModuleList()
        for i in range(len(in_c)):
            self.encoder_sides.append(torch.nn.Sequential(
                module.ocpX2(in_c[i]),
                # conv1x1(side_in_c[i], side_out_c[i]),
                # nn.BatchNorm2d(side_out_c[i]),
                # nn.ReLU(inplace=True)
#                 print("11111111")
            ))

        self.en_ori_side3 = torch.nn.Sequential(
            conv1x1(in_c[5], in_c[4]),
            nn.BatchNorm2d(in_c[4]),
            nn.ReLU(inplace=True)
        )

        self.en_ori_side2 = torch.nn.Sequential(
            conv1x1(in_c[3], in_c[2]),
            nn.BatchNorm2d(in_c[2]),
            nn.ReLU(inplace=True)
        )

        self.en_ori_side1 = torch.nn.Sequential(
            conv1x1(in_c[2], in_c[1]),
            nn.BatchNorm2d(in_c[1]),
            nn.ReLU(inplace=True)
        )

        self.ori_upsample3 = torch.nn.Sequential(
            nn.Conv2d(in_c[4], in_c[3], 3, padding=1, bias=True),
            nn.BatchNorm2d(in_c[3]),
            nn.ReLU(inplace=True),
            nn.Conv2d(in_c[3], in_c[2], 3, padding=1, bias=True),
            nn.BatchNorm2d(in_c[2]),
            nn.ReLU(inplace=True),
#             nn.Upsample(scale_factor=4, mode='bilinear', align_corners=True)
        )
        self.ori_upsample2 = torch.nn.Sequential(
            nn.Conv2d(in_c[3], in_c[2], 3, padding=1, bias=True),
            nn.BatchNorm2d(in_c[2]),
            nn.ReLU(inplace=True),
            nn.Conv2d(in_c[2], in_c[1], 3, padding=1, bias=True),
            nn.BatchNorm2d(in_c[1]),
            nn.ReLU(inplace=True),
            nn.Upsample(scale_factor=2, mode='bilinear', align_corners=True)
        )
        self.ori_upsample1 = torch.nn.Sequential(
            nn.Conv2d(in_c[2], in_c[1], 3, padding=1, bias=True),
            nn.BatchNorm2d(in_c[1]),
            nn.ReLU(inplace=True),
            nn.Conv2d(in_c[1], in_c[0], 3, padding=1, bias=True),
            nn.BatchNorm2d(in_c[0]),
            nn.ReLU(inplace=True),
            nn.Upsample(scale_factor=4, mode='bilinear', align_corners=True)
        )
        self.ori_end = torch.nn.Sequential(
            nn.Conv2d(128, 32, 3, padding=1, bias=True),
            nn.BatchNorm2d(32),
            nn.ReLU(inplace=True),
            module.ocp2(32),
            nn.Conv2d(32, self.args.ori_out_num, 3, padding=1, bias=True)
        )

        self._initialize_weights()

    def forward(self, inputs, save_name='', epoch=-1):
        res_sides = self.res50(inputs)
        features = [self.convolutionX3(inputs)] + res_sides

        ori_xs = self.ori_convolutionX3(inputs)

        if save_name != '':
            self._save_feature([ori_xs] + features, save_name, epoch)

#         ori_encoder_side = []
        for i in range(5, 1, -1):
#             print("i = ",i)
#             encoder_side = self.encoder_sides[i](features[i])
#             exit(0)
            if i == 5:
#                 print("feat[5].shape = ", features[5].shape)
                encoder_side = self.encoder_sides[i](features[i])
#                 print("encoder_side= ",encoder_side.shape)
                ori_encoder_side = self.en_ori_side3(encoder_side)
                ori_encoder_ocp = self.ori_upsample3(ori_encoder_side)
       
            if i == 3:
#                 print("feat[3].shape = ", features[3].shape)
                encoder_side = self.encoder_sides[i](features[i])
                ori_encoder_side = self.en_ori_side2(encoder_side)
#                 ori_encoder_side = self.en_ori_side2(features[i])
                ori_encoder_ocp = module.crop(ori_encoder_ocp, ori_encoder_side)
                ori_encoder_ocp = torch.cat([ori_encoder_ocp, ori_encoder_side], 1)
                ori_encoder_ocp = self.ori_upsample2(ori_encoder_ocp)
     
            if i == 2:
#                 print("aa[2].shape= ", features[2].shape)
                encoder_side = self.encoder_sides[i](features[i])
                ori_encoder_side = self.en_ori_side1(encoder_side)
#                 ori_encoder_side = self.en_ori_side1(features[i])
                ori_encoder_ocp = module.crop(ori_encoder_ocp, ori_encoder_side)
                ori_encoder_ocp = torch.cat([ori_encoder_ocp, ori_encoder_side], 1)
                ori_encoder_ocp = self.ori_upsample1(ori_encoder_ocp)
#                 print("ori_en.shape3=",ori_encoder_ocp.shape)
        
#         print("orixs=",ori_xs.shape)
        ori_encoder_ocp = module.crop(ori_encoder_ocp, ori_xs)
        ori_encoder_ocp = self.ori_end(torch.cat([ori_encoder_ocp, ori_xs], 1))
        n_features = self.pathladder(features)

        edge_outsides = []
        for i in range(len(n_features)):
            edge_outsides.append(self.edge_outsides[i](n_features[i]))
            
        fuse = self.fuse(torch.cat(edge_outsides, 1))
        edge_outsides.append(fuse)

        return edge_outsides, ori_encoder_ocp

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

        if self.res50_model_path is not None:
            if os.path.exists(self.res50_model_path):
                print('File " ', self.res50_model_path, " is exist.")
                pp = torch.load(self.res50_model_path)
                pp.pop("fc.weight")
                pp.pop("fc.bias")
                self.res50.load_state_dict(pp)
            else:
                print('File " ', self.res50_model_path, " is not exist.")
                exit()


def conv3x3(in_planes, out_planes, stride=1):
    """3x3 convolution with padding"""
    return nn.Conv2d(in_planes, out_planes, kernel_size=3, stride=stride,
                     padding=1, bias=True)


def conv1x1(in_planes, out_planes, stride=1):
    """1x1 convolution"""
    return nn.Conv2d(in_planes, out_planes, kernel_size=1, stride=stride, bias=True)
