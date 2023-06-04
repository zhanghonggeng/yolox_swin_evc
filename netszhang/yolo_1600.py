#!/usr/bin/env python3
# -*- coding:utf-8 -*-
# Copyright (c) Megvii, Inc. and its affiliates.

import torch
import torch.nn as nn

from netszhang.swin_darknet_1600 import BaseConv, CSPDarknet, CSPLayer, DWConv
from netszhang.evc_blocks import EVCBlock



class YOLOXHead(nn.Module):
    def __init__(self, num_classes, width=1, in_channels=[128, 256, 512, 1024], act="silu", depthwise=False, ):
        super().__init__()
        Conv = DWConv if depthwise else BaseConv

        self.cls_convs = nn.ModuleList()
        self.reg_convs = nn.ModuleList()
        self.cls_preds = nn.ModuleList()
        self.reg_preds = nn.ModuleList()
        self.obj_preds = nn.ModuleList()
        self.stems = nn.ModuleList()

        self.headsize = 256

        for i in range(len(in_channels)):
            self.stems.append(
                BaseConv(in_channels=int(in_channels[i] * width), out_channels=int(self.headsize * width), ksize=1,
                         stride=1, act=act))
            self.cls_convs.append(nn.Sequential(*[
                Conv(in_channels=int(self.headsize * width), out_channels=int(self.headsize * width), ksize=3, stride=1,
                     act=act),
                Conv(in_channels=int(self.headsize * width), out_channels=int(self.headsize * width), ksize=3, stride=1,
                     act=act),
            ]))
            self.cls_preds.append(
                nn.Conv2d(in_channels=int(self.headsize * width), out_channels=num_classes, kernel_size=1, stride=1,
                          padding=0)
            )

            self.reg_convs.append(nn.Sequential(*[
                Conv(in_channels=int(self.headsize * width), out_channels=int(self.headsize * width), ksize=3, stride=1,
                     act=act),
                Conv(in_channels=int(self.headsize * width), out_channels=int(self.headsize * width), ksize=3, stride=1,
                     act=act)
            ]))
            self.reg_preds.append(
                nn.Conv2d(in_channels=int(self.headsize * width), out_channels=4, kernel_size=1, stride=1, padding=0)
            )
            self.obj_preds.append(
                nn.Conv2d(in_channels=int(self.headsize * width), out_channels=1, kernel_size=1, stride=1, padding=0)
            )

    def forward(self, inputs):
        # ---------------------------------------------------#
        #   inputs输入
        #   P2_out  480, 480, 128
        #   P3_out  240, 240, 256
        #   P4_out  120, 120, 512
        #   P5_out  60, 60, 1024
        # ---------------------------------------------------#
        outputs = []
        for k, x in enumerate(inputs):
            # ---------------------------------------------------#
            #   利用1x1卷积进行通道整合，通道数全变成256
            # ---------------------------------------------------#
            x = self.stems[k](x)
            # ---------------------------------------------------#
            #   利用两个卷积标准化激活函数来进行特征提取
            # ---------------------------------------------------#
            cls_feat = self.cls_convs[k](x)
            # ---------------------------------------------------#
            #   判断特征点所属的种类
            #   80, 80, num_classes
            #   40, 40, num_classes
            #   20, 20, num_classes
            # ---------------------------------------------------#
            cls_output = self.cls_preds[k](cls_feat)

            # ---------------------------------------------------#
            #   利用两个卷积标准化激活函数来进行特征提取
            # ---------------------------------------------------#
            reg_feat = self.reg_convs[k](x)
            # ---------------------------------------------------#
            #   特征点的回归系数
            #   reg_pred 80, 80, 4
            #   reg_pred 40, 40, 4
            #   reg_pred 20, 20, 4
            # ---------------------------------------------------#
            reg_output = self.reg_preds[k](reg_feat)
            # ---------------------------------------------------#
            #   判断特征点是否有对应的物体
            #   obj_pred 80, 80, 1
            #   obj_pred 40, 40, 1
            #   obj_pred 20, 20, 1
            # ---------------------------------------------------#
            obj_output = self.obj_preds[k](reg_feat)

            output = torch.cat([reg_output, obj_output, cls_output], 1)
            outputs.append(output)
        return outputs


class YOLOPAFPN(nn.Module):
    def __init__(self, depth=1, width=1, in_features=("dark2", "dark3", "dark4", "dark5"),
                 in_channels=[128, 256, 512, 1024], depthwise=False, act="silu"):
        super().__init__()
        Conv = DWConv if depthwise else BaseConv
        self.backbone = CSPDarknet(depth, width, depthwise=depthwise, act=act)
        self.in_features = in_features
        self.upsample = nn.Upsample(scale_factor=2, mode="nearest")

        # -------------------------------------------#
        #   60, 60, 1024 -> 60, 60, 512
        # -------------------------------------------#
        self.lateral_conv0 = BaseConv(int(in_channels[3] * width), int(in_channels[2] * width), 1, 1, act=act)
        # -------------------------------------------#
        #   120, 120, 1024 -> 120, 120, 512
        # -------------------------------------------#
        self.C4_p4 = CSPLayer(
            int(2 * in_channels[2] * width),  # 1024
            int(in_channels[2] * width),
            round(3 * depth),
            False,
            depthwise=depthwise,
            act=act,
        )
        # -------------------------------------------#
        #   120, 120, 512 -> 120, 120, 256
        # -------------------------------------------#
        self.reduce_conv1 = BaseConv(int(in_channels[2] * width), int(in_channels[1] * width), 1, 1, act=act)
        # -------------------------------------------#
        #   240, 240, 512 -> 240, 240, 256
        # -------------------------------------------#
        self.C4_p3 = CSPLayer(
            int(2 * in_channels[1] * width),
            int(in_channels[1] * width),
            round(3 * depth),
            False,
            depthwise=depthwise,
            act=act,
        )
        # -------------------------------------------#
        #   240, 240, 256 -> 240, 240, 128
        # -------------------------------------------#
        self.reduce_conv2 = BaseConv(int(in_channels[1] * width), int(in_channels[0] * width), 1, 1, act=act)
        # -------------------------------------------#
        #   480, 480, 256 -> 480, 480, 128            depth = 2
        # -------------------------------------------#
        self.C4_p2 = CSPLayer(
            int(2 * in_channels[0] * width),
            int(in_channels[0] * width),
            round(3 * depth),
            False,
            depthwise=depthwise,
            act=act,
        )

        # -------------------------------------------#
        #   480, 480, 128 -> 240, 240, 128
        # -------------------------------------------#
        self.bu_conv0 = Conv(int(in_channels[0] * width), int(in_channels[0] * width), 3, 2, act=act)
        # -------------------------------------------#
        #   120, 120, 256 -> 120, 120, 256
        # -------------------------------------------#
        self.C4_n3 = CSPLayer(
            int(2 * in_channels[0] * width),
            int(in_channels[1] * width),
            round(3 * depth),
            False,
            depthwise=depthwise,
            act=act,
        )
        # -------------------------------------------#
        #   240, 240, 256 -> 120, 120, 256
        # -------------------------------------------#
        self.bu_conv1 = Conv(int(in_channels[1] * width), int(in_channels[1] * width), 3, 2, act=act)
        # -------------------------------------------#
        #   120, 120, 512 -> 120, 120, 512
        # -------------------------------------------#
        self.C4_n4 = CSPLayer(
            int(2 * in_channels[1] * width),
            int(in_channels[2] * width),
            round(3 * depth),
            False,
            depthwise=depthwise,
            act=act,
        )
        # -------------------------------------------#
        #   120, 120, 512 -> 60, 60, 512
        # -------------------------------------------#
        self.bu_conv2 = Conv(int(in_channels[2] * width), int(in_channels[2] * width), 3, 2, act=act)
        # -------------------------------------------#
        #   60, 60, 1024 -> 60, 60, 1024
        # -------------------------------------------#
        self.C4_n5 = CSPLayer(
            int(2 * in_channels[2] * width),
            int(in_channels[3] * width),
            round(3 * depth),
            False,
            depthwise=depthwise,
            act=act,
        )

        self.evcblock = EVCBlock(
            int(2 * in_channels[1] * width),  #c1
            int(in_channels[2] * width), #c2
            channel_ratio=4, base_channel=16,
            )

        self.epsilon = 1e-4
        self.p3_w1 = nn.Parameter(torch.ones(2, dtype=torch.float32), requires_grad=True)
        self.p3_w1_relu = nn.ReLU()
        self.p4_w1 = nn.Parameter(torch.ones(2, dtype=torch.float32), requires_grad=True)
        self.p4_w1_relu = nn.ReLU()

    def forward(self, input):
        out_features = self.backbone.forward(input)  # dict:4
        [feat1, feat2, feat3, feat4] = [out_features[f] for f in self.in_features]

        # -------------------------------------------#
        #   60, 60, 1024 -> 60, 60, 512
        # -------------------------------------------#
        P5 = self.lateral_conv0(feat4)


        # #-------------------------------------------#
        # #  60, 60, 512 -> 120, 120, 512
        # #-------------------------------------------#
        # P5_upsample = self.upsample(P5)
        # #-------------------------------------------#
        # #  120, 120, 512 + 120, 120, 512 -> 120, 120, 1024
        # #-------------------------------------------#
        # P5_upsample = torch.cat([P5_upsample, feat3], 1)
        # #-------------------------------------------#
        # #   120, 120, 1024 -> 120, 120, 512
        # #-------------------------------------------#
        # P5_upsample = self.C4_p4(P5_upsample)

        # #-------------------------------------------#
        # #   120, 120, 512 -> 120, 120, 256
        # #-------------------------------------------#
        # P4          = self.reduce_conv1(P5_upsample)
        P4 = self.reduce_conv1(self.C4_p4(torch.cat([self.evcblock(self.upsample(P5)), feat3], 1)))

        # #-------------------------------------------#
        # #   120, 120, 256 -> 240, 240, 256
        # #-------------------------------------------#
        # P4_upsample = self.upsample(P4)
        # #-------------------------------------------#
        # #   240, 240, 256 + 240, 240, 256 -> 240, 240, 512
        # #-------------------------------------------#
        # P4_upsample = torch.cat([P4_upsample, feat2], 1)
        # #-------------------------------------------#
        # #   240, 240, 512 -> 240, 240, 256
        # #-------------------------------------------#
        # P4_upsample = self.C4_p3(P4_upsample)
        #
        # #-------------------------------------------#
        # #   240, 240, 256 -> 240, 240, 128
        # #-------------------------------------------#
        # P3          = self.reduce_conv2(P4_upsample)
        P3 = self.reduce_conv2(self.C4_p3(torch.cat([self.upsample(P4), feat2], 1)))
        # print(P3.shape)
        # #-------------------------------------------#
        # #   240, 240, 128 -> 480, 480, 128
        # #-------------------------------------------#
        # P3_upsample = self.upsample(P3)
        # #-------------------------------------------#
        # #   480, 480, 128 + 480, 480, 128 -> 480, 480, 256
        # #-------------------------------------------#
        # P3_upsample = torch.cat([P3_upsample, feat1], 1)
        # #-------------------------------------------#
        # #   480, 480, 256 -> 480, 480, 128
        # #-------------------------------------------#
        # P2_out          = self.C4_p2(P3_upsample)
        P2_out = self.C4_p2(torch.cat([self.upsample(P3), feat1], 1))
        # print(P2_out.shape)
        # #-------------------------------------------#
        # #   480, 480, 128 -> 240, 240, 128
        # #-------------------------------------------#
        # P2_downsample   = self.bu_conv0(P2_out)
        # #-------------------------------------------#
        # #   240, 240, 128 + 240, 240, 128 -> 240, 240, 256
        # #-------------------------------------------#
        # P2_downsample   = torch.cat([P2_downsample, P3], 1)
        # p3_w1           = self.p3_w1_relu(self.p3_w1)
        # weight          = p3_w1 / (torch.sum(p3_w1, dim=0) + self.epsilon)
        # P2_downsample   = weight[0] * feat2 + weight[1] * P2_downsample
        # #-------------------------------------------#
        # #   240, 240, 256 -> 240, 240, 256
        # #-------------------------------------------#
        # P3_out          = self.C4_n3(P2_downsample)
        p3_w1 = self.p3_w1_relu(self.p3_w1)
        weight = p3_w1 / (torch.sum(p3_w1, dim=0) + self.epsilon)
        P3_out = self.C4_n3(weight[0] * feat2 + weight[1] * torch.cat([self.bu_conv0(P2_out), P3], 1))
        # a = weight[0] * feat2 + weight[1] * torch.cat([self.bu_conv0(P2_out), P3], 1)
        # print(P3_out.shape)
        # exit()
        # #-------------------------------------------#
        # #   240, 240, 256 -> 120, 120, 256
        # #-------------------------------------------#
        # P3_downsample   = self.bu_conv1(P3_out)
        # #-------------------------------------------#
        # #   120, 120, 256 + 120, 120, 256 -> 120, 120, 512
        # #-------------------------------------------#
        # P3_downsample   = torch.cat([P3_downsample, P4], 1)
        # p4_w1           = self.p4_w1_relu(self.p4_w1)
        # weight          = p4_w1 / (torch.sum(p4_w1, dim=0) + self.epsilon)
        # P3_downsample   = weight[0] * feat3 + weight[1] * P3_downsample
        # #-------------------------------------------#
        # #   120, 120, 512 -> 120, 120, 512
        # #-------------------------------------------#
        # P4_out          = self.C4_n4(P3_downsample)
        p4_w1 = self.p4_w1_relu(self.p4_w1)
        weight = p4_w1 / (torch.sum(p4_w1, dim=0) + self.epsilon)
        P4_out = self.C4_n4(weight[0] * feat3 + weight[1] * torch.cat([self.bu_conv1(P3_out), P4], 1))
        # print(P4_out.shape)

        # #-------------------------------------------#
        # #   120, 120, 512 -> 60, 60, 512
        # #-------------------------------------------#
        # P4_downsample   = self.bu_conv2(P4_out)
        # #-------------------------------------------#
        # #   60, 60, 512 + 60, 60, 512 -> 60, 60, 1024
        # #-------------------------------------------#
        # P4_downsample   = torch.cat([P4_downsample, P5], 1)
        # #-------------------------------------------#
        # #   60, 60, 1024 -> 60, 60, 1024
        # #-------------------------------------------#
        # P5_out          = self.C4_n5(P4_downsample)
        P5_out = self.C4_n5(torch.cat([self.bu_conv2(P4_out), P5], 1))
        # print(P5_out.shape)
        # exit()
        return (P2_out, P3_out, P4_out, P5_out)


class YoloBody(nn.Module):
    def __init__(self, num_classes, phi):
        super().__init__()
        depth_dict = {'nano': 0.33, 'tiny': 0.33, 's': 0.33, 'm': 0.67, 'l': 1.00, 'x': 1.33, }
        width_dict = {'nano': 0.25, 'tiny': 0.375, 's': 0.50, 'm': 0.75, 'l': 1.00, 'x': 1.25, }
        depth, width = depth_dict[phi], width_dict[phi]
        depthwise = True if phi == 'nano' else False
        self.backbone = YOLOPAFPN(depth, width, depthwise=depthwise)
        self.head = YOLOXHead(num_classes, width, depthwise=depthwise)

    def forward(self, x):  # bc,3,1920,1920
        fpn_outs = self.backbone.forward(x)
        outputs = self.head.forward(fpn_outs)
        # return fpn_outs
        return outputs

if __name__== '__main__':
    model = YoloBody(10, 'l')
    test_tensor = torch.randn([1, 3, 1280, 1280])
    out = model.forward(test_tensor)
    for i in out:
        print(i.shape)
