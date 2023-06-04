#!/usr/bin/env python3
# -*- coding:utf-8 -*-
# Copyright (c) Megvii, Inc. and its affiliates.

import torch
import torch.nn as nn

from nets_swin_evc.swin_darknet import BaseConv, CSPDarknet, CSPLayer, DWConv
from nets_swin_evc.evc_blocks import EVCBlock


class YOLOXHead(nn.Module):
    # 四个头 in_channels=[256, 512, 1024] 改成 in_channels=[128, 256, 512, 1024]
    def __init__(self, num_classes, width=1, in_channels=[128, 256, 512, 1024], act="silu", depthwise=False, ):
        super().__init__()
        Conv = DWConv if depthwise else BaseConv

        self.cls_convs = nn.ModuleList()
        self.reg_convs = nn.ModuleList()
        self.cls_preds = nn.ModuleList()
        self.reg_preds = nn.ModuleList()
        self.obj_preds = nn.ModuleList()
        self.stems = nn.ModuleList()

        for i in range(len(in_channels)):
            self.stems.append(
                BaseConv(in_channels=int(in_channels[i] * width), out_channels=int(256 * width), ksize=1, stride=1,
                         act=act))
            self.cls_convs.append(nn.Sequential(*[
                Conv(in_channels=int(256 * width), out_channels=int(256 * width), ksize=3, stride=1, act=act),
                Conv(in_channels=int(256 * width), out_channels=int(256 * width), ksize=3, stride=1, act=act),
            ]))
            self.cls_preds.append(
                nn.Conv2d(in_channels=int(256 * width), out_channels=num_classes, kernel_size=1, stride=1, padding=0)
            )

            self.reg_convs.append(nn.Sequential(*[
                Conv(in_channels=int(256 * width), out_channels=int(256 * width), ksize=3, stride=1, act=act),
                Conv(in_channels=int(256 * width), out_channels=int(256 * width), ksize=3, stride=1, act=act)
            ]))
            self.reg_preds.append(
                nn.Conv2d(in_channels=int(256 * width), out_channels=4, kernel_size=1, stride=1, padding=0)
            )
            self.obj_preds.append(
                nn.Conv2d(in_channels=int(256 * width), out_channels=1, kernel_size=1, stride=1, padding=0)
            )

    def forward(self, inputs):
        # ---------------------------------------------------#
        #   inputs输入
        #   P3_out  80, 80, 256
        #   P4_out  40, 40, 512
        #   P5_out  20, 20, 1024
        # ---------------------------------------------------#
        outputs = []
        for k, x in enumerate(inputs):
            # ---------------------------------------------------#
            #   利用1x1卷积进行通道整合
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
    # 四个头 in_features=("dark2", "dark3", "dark4", "dark5"), in_channels=[128, 256, 512, 1024]
    def __init__(self, depth=1, width=1, in_features=("dark2", "dark3", "dark4", "dark5"),
                 in_channels=[128, 256, 512, 1024],
                 depthwise=False, act="silu"):
        super().__init__()
        Conv = DWConv if depthwise else BaseConv
        self.backbone = CSPDarknet(depth, width, depthwise=depthwise, act=act)
        self.in_features = in_features

        self.upsample = nn.Upsample(scale_factor=2, mode="nearest")

        # -------------------------------------------#
        #   [384,20,20] -> [192,20,20]  四个头输入通道2 1变成32
        # -------------------------------------------#
        self.lateral_conv0 = BaseConv(int(in_channels[3] * width), int(in_channels[2] * width), 1, 1, act=act)
        # print("width", width)
        # print("lateral_conv0", self.lateral_conv0)
        # -------------------------------------------#
        #   [192,40,40] -> [384,40,40] 四个头 输入通道 1 1 变成 2 2
        # -------------------------------------------#
        # C4表示 feet4 P4表示算德P4
        self.C4_p4 = CSPLayer(
            int(2 * in_channels[2] * width),
            int(in_channels[2] * width),
            round(3 * depth),
            False,
            depthwise=depthwise,
            act=act,
        )

        # -------------------------------------------#
        #   四个头输入通道 1 0 改成 2 1
        # -------------------------------------------#
        self.reduce_conv1 = BaseConv(int(in_channels[2] * width), int(in_channels[1] * width), 1, 1, act=act)
        # print("reduce_conv1", self.reduce_conv1)
        # -------------------------------------------#
        #   四个头输入通道 0 0 改成 1 1
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
        #   四个头输入通道 1 0
        # -------------------------------------------#
        self.reduce_conv2 = BaseConv(int(in_channels[1] * width), int(in_channels[0] * width), 1, 1, act=act)
        # print("reduce_conv2", self.reduce_conv2)
        # -------------------------------------------#
        #   四个头输入通道 0 0
        # -------------------------------------------#
        self.C4_p2 = CSPLayer(
            int(2 * in_channels[0] * width),
            int(in_channels[0] * width),
            round(3 * depth),
            False,
            depthwise=depthwise,
            act=act,
        )

        # =============================================
        self.bu_conv0 = Conv(int(in_channels[0] * width), int(in_channels[0] * width), 3, 2, act=act)
        # ============================================

        self.C4_n3 = CSPLayer(
            int(2 * in_channels[0] * width),
            int(in_channels[1] * width),
            round(3 * depth),
            False,
            depthwise=depthwise,
            act=act,
        )
        # -------------------------------------------#
        #   四个头输入通道 2  2
        # -------------------------------------------#
        self.bu_conv2 = Conv(int(in_channels[2] * width), int(in_channels[2] * width), 3, 2, act=act)

        # -------------------------------------------#
        #   四个头输入通道 0 1 改成 1 2
        # -------------------------------------------#
        self.evcblock = EVCBlock(
            int(2 * in_channels[1] * width),  # c1
            int(in_channels[2] * width),  # c2
            channel_ratio=4, base_channel=16,
        )
        # -------------------------------------------#
        #   四个头输入通道 0 1
        # -------------------------------------------#
        self.C3_n3 = CSPLayer(
            int(2 * in_channels[0] * width),
            int(in_channels[1] * width),
            round(3 * depth),
            False,
            depthwise=depthwise,
            act=act,
        )

        # -------------------------------------------#
        #   四个头输入通道 1  1
        # -------------------------------------------#
        self.bu_conv1 = Conv(int(in_channels[1] * width), int(in_channels[1] * width), 3, 2, act=act)
        # -------------------------------------------#
        #   四个头输入通道  1 2
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
        #   四个头输入通道  2 3
        # -------------------------------------------#
        self.C4_n5 = CSPLayer(
            int(2 * in_channels[2] * width),
            int(in_channels[3] * width),
            round(3 * depth),
            False,
            depthwise=depthwise,
            act=act,
        )

    def forward(self, input):
        out_features = self.backbone.forward(input)
        # dark2  dark3  dark4  dark5
        # [48,160,160] [96,80,80] [192,40,40] [384,20,20]
        [feat1, feat2, feat3, feat4] = [out_features[f] for f in self.in_features]
        # -------------- -----------------------------#
        #   [384,20,20] -> [192,20,20]
        # -------------------------------------------#
        P5 = self.lateral_conv0(feat4)
        # -------------- -----------------------------#
        #   [192,20,20] -> [96,40,40]
        # -------------------------------------------#
        P4 = self.reduce_conv1(self.C4_p4(torch.cat([self.evcblock(self.upsample(P5)), feat3], 1)))
        # -------------- -----------------------------#
        #  [96,40,40] -> [48,80,80]
        # -------------------------------------------#
        P3 = self.reduce_conv2(self.C4_p3(torch.cat([self.upsample(P4), feat2], 1)))
        # -------------- -----------------------------#
        #  [48,80,80] -> [48,160,160]
        # -------------------------------------------#
        P2_out = self.C4_p2(torch.cat([self.upsample(P3), feat1], 1))
        # -------------- -----------------------------#
        #  [48,160,160] -> [96,80,80]
        # -------------------------------------------#
        P3_out = self.C4_n3(torch.cat([self.bu_conv0(P2_out), P3], 1))
        # -------------- -----------------------------#
        #  [96,80,80] -> [192,40,40]
        # -------------------------------------------#
        P4_out = self.C4_n4(torch.cat([self.bu_conv1(P3_out), P4], 1))
        # -------------- -----------------------------#
        #  [192,40,40] -> [384,20,20]
        # -------------------------------------------#
        P5_out = self.C4_n5(torch.cat([self.bu_conv2(P4_out), P5], 1))

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

    def forward(self, x):
        fpn_outs = self.backbone.forward(x)
        outputs = self.head.forward(fpn_outs)
        return outputs


if __name__ == "__main__":
    model = YoloBody(15, 'l')
    # print(model)
    #
    tensor_test = torch.randn([1, 3, 1024, 1024])
    out = model.forward(tensor_test)
    for i in out:
        print(i.shape)
