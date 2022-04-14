#!/usr/bin/env python
# -*- coding: utf-8 -*-

# # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # 
# @Project: BraTS2020ACMINetCrossValidationCode
# @IDE: PyCharm
# @Author: ZhuangYuZhou
# @E-mail: 605540375@qq.com
# @Time: 2022/3/12
# @Desc: 
# # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # #


import torch
import torch.nn as nn
import torch.nn.functional as F

class SIGR(nn.Module):
    """
    Spatial Interaction Graph Reasoning
    """

    def __init__(self, planes):
        super(SIGR, self).__init__()

        # self.spatial_num_node = 8x(DWH)/2
        self.spatial_num_state = planes // 2

        self.downsampling = nn.Sequential(
            nn.Conv3d(planes, planes, 3, groups=planes, stride=2, padding=1, bias=False),
            nn.GroupNorm(num_groups=16, num_channels=planes),
        )

        self.node_k = nn.Conv3d(planes, self.spatial_num_state, kernel_size=1)
        self.node_v = nn.Conv3d(planes, self.spatial_num_state, kernel_size=1)
        self.node_q = nn.Conv3d(planes, self.spatial_num_state, kernel_size=1)

        self.conv_wg = nn.Conv1d(self.spatial_num_state, self.spatial_num_state, kernel_size=1, bias=False)
        self.bn_wg = nn.GroupNorm(num_groups=16,num_channels=self.spatial_num_state)
        self.softmax = nn.Softmax(dim=2)

        self.out = nn.Sequential(nn.Conv3d(self.spatial_num_state, planes, kernel_size=1),
                                  nn.GroupNorm(num_groups=16,num_channels=planes))

    def forward(self, input_feat):

        # # # # # # # # # # # # # # # # #
        # Projection
        # # # # # # # # # # # # # # # # #

        # # # # #
        # downsampling
        # # # # #
        # (C,D,H,W) -> (C, D/2, H/2, W/2)
        x = self.downsampling(input_feat)

        # V_s
        node_v = self.node_v(x)
        # Q_s
        node_q = self.node_q(x)
        # K_s
        node_k = self.node_k(x)

        b, c, d, h, w = node_v.size()
        # print('node_v.size()',node_v.size())

        # # # # #
        # reshape
        # # # # #
        # V_s
        node_v = node_v.view(b, c, -1)
        # Q_s
        node_q = node_q.view(b, c, -1)
        # K_s
        node_k = node_k.view(b, c, -1)

        # # # # #
        # transpose
        # # # # #
        # (V_s)^T: (DHW/8)×C/2
        node_v = node_v.permute(0, 2, 1)
        # Q_s: C/2×(DHW/8)
        node_q = node_q
        # (K_s)^T: (DHW/8)×C/2
        node_k = node_k.permute(0, 2, 1)

        # print(node_v.size(), node_q.size(), node_k.size())

        # # # # # # # # # # # # # # # # #
        # Reasoning
        # # # # # # # # # # # # # # # # #

        # # # # #
        # graph convolution
        # # # #
        # A_s = sfm(K_s x Q_s): (N_s, N_s) (DHW/8)×(DHW/8)
        A = self.softmax(torch.bmm(node_k,node_q))
        # A(V)^T = (DHW/8)×C/2
        AV = torch.bmm(A, node_v)
        AV = AV.transpose(1, 2).contiguous()
        AVW = self.conv_wg(AV)
        #  theta(AVW)： C/2×(DHW/8)
        AVW = F.relu_(self.bn_wg(AVW))

        # # # # # # # # # # # # # # # # #
        # Reprojection
        # # # # # # # # # # # # # # # # #
        AVW = AVW.view(b, c, d, h, -1)
        sigr_out = self.out(AVW) + x


        F_sg = F.interpolate(sigr_out, size=input_feat.size()[2:], mode='trilinear', align_corners=False)

        # spatial gr output
        spatial_gr_out = F_sg + input_feat

        return spatial_gr_out

class FIGR(nn.Module):
    """
    Feature Interaction Graph Reasoning

    """

    def __init__(self, planes, ratio=4):
        super(FIGR, self).__init__()

        self.feature_num_node = planes // ratio
        self.feature_num_state = planes // ratio * 2

        self.phi = nn.Conv3d(planes, self.feature_num_state, kernel_size=1, bias=False)
        self.bn_phi = nn.GroupNorm(num_groups=16, num_channels=self.feature_num_state)
        self.theta = nn.Conv3d(planes, self.feature_num_node, kernel_size=1, bias=False)
        self.bn_theta = nn.GroupNorm(num_groups=16, num_channels=planes // ratio)

        #  Interaction Space
        #  Adjacency Matrix: (-)A_g
        self.conv_adj = nn.Conv1d(self.feature_num_node, self.feature_num_node, kernel_size=1, bias=False)
        self.bn_adj = nn.GroupNorm(num_groups=16, num_channels=planes // ratio)

        #  State Update Function: W_g
        self.conv_wg = nn.Conv1d(self.feature_num_state, self.feature_num_state, kernel_size=1, bias=False)
        self.bn_wg = nn.GroupNorm(num_groups=16, num_channels=self.feature_num_state)

        #  last fc
        self.conv3 = nn.Conv3d(self.feature_num_state, planes, kernel_size=1, bias=False)
        self.bn3 = nn.GroupNorm(num_groups=16, num_channels=planes)

    def to_matrix(self, x):
        n, c, d, h, w = x.size()
        x = x.view(n, c, -1)
        return x

    def forward(self, input_feat):
        # channel gr
        # # # # Projection Space # # # #
        x_sqz, b = input_feat, input_feat

        x_sqz = self.phi(x_sqz)
        x_sqz = self.bn_phi(x_sqz)
        x_sqz = self.to_matrix(x_sqz)

        b = self.theta(b)
        b = self.bn_theta(b)
        b = self.to_matrix(b)
        # Project
        z_idt = torch.matmul(x_sqz, b.transpose(1, 2))

        # # # # Interaction Space # # # #
        z = z_idt.transpose(1, 2).contiguous()
        # print('z',z.size())

        z = self.conv_adj(z)
        z = self.bn_adj(z)
        # print('z',z.size())
        # assert False
        z = z.transpose(1, 2).contiguous()
        # Laplacian smoothing: (I - A_g)Z => Z - A_gZ
        z += z_idt

        z = self.conv_wg(z)
        z = F.relu_(self.bn_wg(z))

        # # # # Re-projection Space # # # #
        # Re-project
        y = torch.matmul(z, b)
        n, _, d, h, w= input_feat.size()
        y = y.view(n, -1, d, h, w)
        y = self.conv3(y)
        F_fg = self.bn3(y)

        channel_gr_feat = input_feat + F_fg
        return channel_gr_feat

class VDIGR(nn.Module):
    """
        VolumetricDualInteractionGraphReasoning
    """
    def __init__(self, planes, ratio=4, fusion=False):
        super(VDIGR, self).__init__()

        self.spatial_graph_reasoning_module = SIGR(planes=planes)
        self.feature_graph_reasoning_module = FIGR(planes=planes, ratio=ratio)

        self.fusion = fusion
        if fusion:
            self.final = nn.Sequential(
                                       nn.Conv3d(planes * 2, planes, kernel_size=3, stride=1, padding=1, bias=False),
                                       nn.GroupNorm(num_groups=16,num_channels=planes),
                                       nn.ReLU(inplace=True)
                                       )



    def forward(self, feat):
        spatial_gr_feat = self.spatial_graph_reasoning_module(feat)
        # print('spatial_gr_feat.shape',spatial_gr_feat.shape)

        feature_gr_feat = self.feature_graph_reasoning_module(feat)
        # print('feature_gr_feat.shape', feature_gr_feat.shape)


        out = torch.cat((spatial_gr_feat, feature_gr_feat),dim=1)

        if self.fusion:
            out = self.final(out)

        return out

def count_param(model):
    param_count = 0
    for param in model.parameters():
        param_count += param.view(-1).size()[0]
    return param_count

if __name__ == '__main__':
    normalize = True
    input_tensor = torch.randn(2, 128, 16, 16, 16)
    print('input_tensor', input_tensor.shape)

    net = VDIGR(planes=input_tensor.shape[1], fusion=False)

    param = count_param(net)
    print('net totoal parameters: %.2fM (%d)' % (param / 1e6, param))

    output_tensor = net(input_tensor)

    print('output_tensor', output_tensor.shape)

