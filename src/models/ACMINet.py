#!/usr/bin/env python
# -*- coding: utf-8 -*-

# # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # 
# @Project: ACMINetGraphReasoning
# @IDE: PyCharm
# @Author: ZhuangYuZhou
# @E-mail: 605540375@qq.com
# @Time: 2022/1/7
# @Desc: 
# # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # #

import torch
from torch import nn
from torch.utils.checkpoint import checkpoint_sequential
from torch.nn import functional as F
from src.models.layers import ConvBnRelu, UBlock, conv1x1
from src.models.graph_lib import VDIGR


# # # # # # # # # # # # # # # # # # # # # # # # #
# feature alignment
# # # # # # # # # # # # # # # # # # # # # # # # #
class Aligned3DUpsampleConcat(nn.Module):
    def __init__(self, features):
        super(Aligned3DUpsampleConcat, self).__init__()
        assert features % 16 == 0, 'base 16 filters'

        self.upsample = nn.Upsample(scale_factor=2, mode="trilinear", align_corners=False)

        self.delta_flow_gen = nn.Sequential(
            nn.Conv3d(features * 2, features, kernel_size=1, bias=False),
            nn.GroupNorm(num_groups=16, num_channels=features),
            # volumetric out feat: 3 grid
            nn.Conv3d(features, 3, kernel_size=3, padding=1)
        )

        self.aligned_and_upsample_merge_conv = nn.Sequential(
            nn.Conv3d(features * 2, features, kernel_size=1),
        )

    def tilinear_interpolate_torch_gridsample(self, input, size, delta=0):
        out_d, out_h, out_w = size

        n, c, d, h, w = input.shape
        s = 1.0

        # normal
        norm = torch.tensor([[[[[d / s, h / s, w / s]]]]]).type_as(input).to(input.device)

        d_list = torch.linspace(-1.0, 1.0, out_d)
        h_list = torch.linspace(-1.0, 1.0, out_h)
        w_list = torch.linspace(-1.0, 1.0, out_w)

        d_list, h_list, w_list = torch.meshgrid(d_list, h_list, w_list)
        grid = torch.cat([w_list.unsqueeze(3), h_list.unsqueeze(3), d_list.unsqueeze(3)], dim=3)

        # n, d, h, w, c
        grid = grid.repeat(n, 1, 1, 1, 1).type_as(input).to(input.device)

        # n, d, h, w, c
        delta_permute = delta.permute(0, 2, 3, 4, 1)

        grid = grid + delta_permute / norm

        output = F.grid_sample(input, grid, align_corners=False)
        return output

    def forward(self, high_resolution, low_resolution):
        high_resolution_d, high_resolution_h, high_resolution_w = high_resolution.size(2), high_resolution.size(
            3), high_resolution.size(4)
        low_resolution_d, low_resolution_h, low_resolution_w = low_resolution.size(2), low_resolution.size(
            3), low_resolution.size(4)

        assert low_resolution_d == high_resolution_d // 2 and low_resolution_h == high_resolution_h // 2 and low_resolution_w == high_resolution_w // 2

        low_stage = high_resolution
        high_stage = low_resolution

        d, h, w = low_stage.size(2), low_stage.size(3), low_stage.size(4)

        # upscale
        high_stage = self.upsample(high_stage)

        concat = torch.cat((low_stage, high_stage), 1)

        # error back propagation  to delta_gen
        delta_flow = self.delta_flow_gen(concat)

        # split
        high_stage_aligned = self.tilinear_interpolate_torch_gridsample(high_stage, (d, h, w), delta_flow)
        high_stage_upsample = high_stage

        low_stage_final = low_stage
        high_stage_final = self.aligned_and_upsample_merge_conv(torch.cat((high_stage_aligned, high_stage_upsample), 1))

        # concat
        output_alignment_tensor = torch.cat([low_stage_final, high_stage_final], dim=1)

        return output_alignment_tensor

class CrossModalityBranchAttention(nn.Module):
    def __init__(self, num_channels, reduction_ratio=2):
        super(CrossModalityBranchAttention, self).__init__()
        # 指定output size
        self.global_avg_pool = nn.AdaptiveAvgPool3d(output_size=(1, 1, 1))
        self.global_max_pool = nn.AdaptiveMaxPool3d(output_size=(1, 1, 1))

        num_channels_reduced = num_channels // reduction_ratio
        self.reduction_ratio = reduction_ratio

        self.fc1 = nn.Linear(num_channels, num_channels_reduced, bias=True)

        self.fc2_branch_1 = nn.Linear(num_channels_reduced, num_channels, bias=True)
        self.fc2_branch_2 = nn.Linear(num_channels_reduced, num_channels, bias=True)

        # self.relu = nn.ReLU(inplace=True)
        # self.sigmoid = nn.Sigmoid()
        self.softmax = nn.Softmax(dim=1)

    def forward(self, input_tensor):
        """
        :param input_tensor: X, shape = (batch_size, num_channels, D, H, W)
        :return: output tensor: (batch_size, num_channels, 1, 1, 1)
        """
        batch_size, num_channels, D, H, W = input_tensor.size()

        # Average Pooling along each channel
        avg_squeeze_tensor = self.global_avg_pool(input_tensor)

        # Max Pooling along each channel
        max_squeeze_tensor = self.global_max_pool(input_tensor)

        # channel excitation
        # avg
        avg_fc_out_1 = self.fc1(avg_squeeze_tensor.view(batch_size, num_channels))
        # max
        max_fc_out_1 = self.fc1(max_squeeze_tensor.view(batch_size, num_channels))

        #
        shared_fc_out_1 = avg_fc_out_1 + max_fc_out_1

        branch_1_attention_weight = self.fc2_branch_1(shared_fc_out_1)

        branch_2_attention_weight = self.fc2_branch_2(shared_fc_out_1)

        branch_1_attention_weight = self.softmax(branch_1_attention_weight)
        branch_2_attention_weight = self.softmax(branch_2_attention_weight)

        # reshape
        branch_1_ca_weight = branch_1_attention_weight.view(batch_size, num_channels, 1, 1, 1)
        branch_2_ca_weight = branch_2_attention_weight.view(batch_size, num_channels, 1, 1, 1)

        return branch_1_ca_weight, branch_2_ca_weight

class CrossModalityInterConvModule(nn.Module):
    def __init__(self, num_channels, norm_layer, dropout, dilation=(1, 1), reduction_ratio=2):
        super(CrossModalityInterConvModule, self).__init__()
        self.modality_branch_channels = num_channels // 2
        self.branch_att_weight_func = CrossModalityBranchAttention(num_channels=num_channels,
                                                                reduction_ratio=reduction_ratio)

        self.modality_branch_1_conv = ConvBnRelu(self.modality_branch_channels, self.modality_branch_channels,
                                                 norm_layer, dilation=dilation[0], dropout=dropout)
        self.modality_branch_2_conv = ConvBnRelu(self.modality_branch_channels, self.modality_branch_channels,
                                                 norm_layer, dilation=dilation[0], dropout=dropout)

        self.modality_fusion_conv = ConvBnRelu(3 * num_channels, num_channels, norm_layer, dilation=dilation[1],
                                               dropout=dropout)

    def forward(self, input_tensor):

        # # # # # # # # # # # # # # # # # # #
        # Grouping
        # # # # # # # # # # # # # # # # # # #
        x = input_tensor

        # grouping
        input_modality_branch_1_tensor, input_modality_branch_2_tensor = torch.chunk(input_tensor, 2, dim=1)

        # # # # # # # # # # # # # # # # # # #
        # Interaction
        # # # # # # # # # # # # # # # # # # #
        # attention weight
        branch_1_channel_weight, branch_2_channel_weight = self.branch_att_weight_func(input_tensor)

        conv_modality_branch_1_tensor = self.modality_branch_1_conv(input_modality_branch_1_tensor)
        conv_modality_branch_2_tensor = self.modality_branch_2_conv(input_modality_branch_2_tensor)

        concat_modality_branch_1_tensor = torch.cat([conv_modality_branch_1_tensor, input_modality_branch_2_tensor],
                                                    dim=1)
        concat_modality_branch_2_tensor = torch.cat([conv_modality_branch_2_tensor, input_modality_branch_1_tensor],
                                                    dim=1)

        recalibration_modality_branch_1_feature = torch.mul(concat_modality_branch_1_tensor, branch_1_channel_weight)
        recalibration_modality_branch_2_feature = torch.mul(concat_modality_branch_2_tensor, branch_2_channel_weight)

        # # # # # # # # # # # # # # # # # # #
        # Fusion
        # # # # # # # # # # # # # # # # # # #
        concat_tensor = torch.cat([recalibration_modality_branch_1_feature, recalibration_modality_branch_2_feature, x],
                                  dim=1)

        output_tensor = self.modality_fusion_conv(concat_tensor)

        return output_tensor

from collections import OrderedDict

class CrossModalityInterConv(nn.Sequential):
    """Unet mainstream downblock.
    """

    def __init__(self, inplanes, outplanes, norm_layer, dilation=(1, 1), dropout=0):
        super(CrossModalityInterConv, self).__init__(
            OrderedDict(
                [
                    ('ConvBnRelu1',
                     CrossModalityInterConvModule(num_channels=inplanes, norm_layer=norm_layer, dilation=dilation,
                                                  dropout=dropout, reduction_ratio=2)),
                    ('ConvBnRelu2', ConvBnRelu(inplanes, outplanes, norm_layer, dilation=dilation[1], dropout=dropout)),
                ]
            )

        )

class MultiModalFeatureExtraction(nn.Module):
    def __init__(self, modal_num, base_n_filter):
        super(MultiModalFeatureExtraction, self).__init__()
        self.modal_num = modal_num
        self.base_n_filter = base_n_filter
        self.conv_filter = base_n_filter//modal_num
        self.conv_list =nn.ModuleList()
        for i in range(modal_num):
            self.conv_list.append(nn.Sequential(
                nn.Conv3d(1, self.base_n_filter, kernel_size=(3,3,3), stride=(1,1,1), padding=(1,1,1), bias=False),
                nn.Conv3d(self.base_n_filter, self.conv_filter, kernel_size=(1, 1, 1), stride=(1, 1, 1), padding=(0, 0, 0), bias=False),
                # nn.InstanceNorm3d(self.conv_filter),
                # nn.ReLU(inplace=True)
                #
            ))




    def forward(self, input_tensor):
        # input_tensor = x
        x = input_tensor

        modal_input_list = torch.chunk(input_tensor, self.modal_num, dim=1)
        modal_feature_list = []
        # the part of split
        for i, (current_conv, current_input)in enumerate(zip(self.conv_list, modal_input_list)):

            modal_feature_list.append(current_conv(current_input))

        output_tensor = torch.cat(modal_feature_list, dim=1)

        return output_tensor


class ACMINet(nn.Module):
    name = "ACMINet"

    def __init__(self, inplanes, num_classes, width, norm_layer=None, deep_supervision=False, dropout=0, **kwargs):
        super(ACMINet, self).__init__()
        # width:32 -> [32, 64, 128, 256]
        features = [width * 2 ** i for i in range(4)]
        print(features)

        self.deep_supervision = deep_supervision

        self.multimodal_conv = nn.Sequential()
        self.multimodal_conv.add_module('cross_modal_fusion_conv',
                                        MultiModalFeatureExtraction(modal_num=inplanes, base_n_filter=features[0]))
        self.multimodal_conv.add_module('cross_modal_interaction_module1',
                                        CrossModalityInterConvModule(num_channels=features[0],
                                                                     norm_layer=get_norm_layer(), dilation=(1, 1),
                                                                     dropout=dropout, reduction_ratio=2))

        self.encoder1 = ConvBnRelu(features[0], features[0], norm_layer, dropout=dropout)
        self.encoder2 = CrossModalityInterConv(features[0], features[1], norm_layer, dropout=dropout)
        self.encoder3 = CrossModalityInterConv(features[1], features[2], norm_layer, dropout=dropout)
        self.encoder4 = CrossModalityInterConv(features[2], features[3], norm_layer, dropout=dropout)

        self.bottom = nn.Sequential(ConvBnRelu(features[3], features[3], norm_layer, dropout=dropout),
                                    VDIGR(features[3],ratio=4, fusion=False),
                                    UBlock(features[3]*2, features[3], features[3], norm_layer, dropout=dropout)
                                    )

        self.bottom_2 = ConvBnRelu(features[3] * 2, features[2], norm_layer, dropout=dropout)

        self.downsample = nn.MaxPool3d(2, 2)

        self.decoder3 = UBlock(features[2] * 2, features[2], features[1], norm_layer, dropout=dropout)
        self.decoder2 = UBlock(features[1] * 2, features[1], features[0], norm_layer, dropout=dropout)
        self.decoder1 = UBlock(features[0] * 2, features[0], features[0], norm_layer, dropout=dropout)

        self.upsample = nn.Upsample(scale_factor=2, mode="trilinear", align_corners=False)

        self.align_upsample_dec3 = Aligned3DUpsampleConcat(features=features[2])

        self.align_upsample_dec2 = Aligned3DUpsampleConcat(features=features[1])

        self.align_upsample_dec1 = Aligned3DUpsampleConcat(features=features[0])

        self.outconv = conv1x1(features[0], num_classes)

        if self.deep_supervision:
            self.deep_bottom = nn.Sequential(
                conv1x1(features[3], num_classes),
                nn.Upsample(scale_factor=8, mode="trilinear", align_corners=False))

            self.deep_bottom2 = nn.Sequential(
                conv1x1(features[2], num_classes),
                nn.Upsample(scale_factor=8, mode="trilinear", align_corners=False))

            self.deep3 = nn.Sequential(
                conv1x1(features[1], num_classes),
                nn.Upsample(scale_factor=4, mode="trilinear", align_corners=False))

            self.deep2 = nn.Sequential(
                conv1x1(features[0], num_classes),
                nn.Upsample(scale_factor=2, mode="trilinear", align_corners=False))

        self._init_weights()

    def _init_weights(self):
        for m in self.modules():
            if isinstance(m, (nn.Conv3d, nn.Conv2d, nn.Conv1d)):
                nn.init.kaiming_normal_(m.weight, mode='fan_out', nonlinearity='relu')
            elif isinstance(m, (nn.BatchNorm3d, nn.GroupNorm, nn.InstanceNorm3d)):
                nn.init.constant_(m.weight, 1)
                nn.init.constant_(m.bias, 0)

    def forward(self, x):
        # t1, t1ce, t2, flair
        # Cross-Modality Interaction Encoder
        x = self.multimodal_conv(x)

        down1 = self.encoder1(x)

        down2 = self.downsample(down1)

        down2 = self.encoder2(down2)

        down3 = self.downsample(down2)

        down3 = self.encoder3(down3)

        down4 = self.downsample(down3)

        # Bottleneck
        down4 = self.encoder4(down4)
        bottom = self.bottom(down4)
        bottom_2 = self.bottom_2(torch.cat([down4, bottom], dim=1))

        # Aligned Decoder
        up3_align_concat_tensor = self.align_upsample_dec3(high_resolution=down3, low_resolution=bottom_2)

        up3_align_concat_tensor = self.decoder3(up3_align_concat_tensor)

        up3 = up3_align_concat_tensor

        up2_align_concat_tensor = self.align_upsample_dec2(high_resolution=down2,
                                                           low_resolution=up3_align_concat_tensor)

        up2_align_concat_tensor = self.decoder2(up2_align_concat_tensor)

        up2 = up2_align_concat_tensor

        up1_align_concat_tensor = self.align_upsample_dec1(high_resolution=down1,
                                                           low_resolution=up2_align_concat_tensor)

        up1_align_concat_tensor = self.decoder1(up1_align_concat_tensor)

        out = self.outconv(up1_align_concat_tensor)

        if self.deep_supervision:
            deeps = []
            for seg, deep in zip(
                    [down4, bottom_2, up3, up2],
                    [self.deep_bottom, self.deep_bottom2, self.deep3, self.deep2]):
                deeps.append(deep(seg))
            return out, deeps

        return out

from src.models.layers import get_norm_layer, count_param

# net
if __name__ == '__main__':
    n_modal = 4
    n_classes = 4
    deep_supervision = True

    # ACMINet
    net = ACMINet(inplanes=n_modal, num_classes=n_classes, width=32, norm_layer=get_norm_layer(),
                      deep_supervision=deep_supervision, dropout=0)

    param = count_param(net)
    print('net totoal parameters: %.2fM (%d)' % (param / 1e6, param))

    ## "brats2020":  {'bg':0, "WT": 1,  'TC': 1, 'ET': 1}
    net.eval()
    with torch.no_grad():
        input_tensor = torch.rand(1, n_modal, 64, 64, 64)

        # Param and FLOPs
        from thop import profile
        flops, params = profile(net, inputs=(input_tensor,))
        print('net.name', net.name)
        print('FLOPs = ' + str(flops / 1000 ** 3) + 'G')
        print('Params = ' + str(params / 1000 ** 2) + 'M')

        if deep_supervision:
            seg_output, deep_sups = net(input_tensor)
            for deep_idx, deep_sup in enumerate(deep_sups):
                print('deep_idx:%s, deep_sup.size: %s' % (deep_idx, str(deep_sup.size())))

        else:
            seg_output = net(input_tensor)
            print(seg_output.shape)
