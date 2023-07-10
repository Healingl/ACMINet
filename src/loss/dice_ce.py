#!/usr/bin/env python
# -*- coding: utf-8 -*-

# # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # 
# @Project: BrainTS2020SegDeepSupTorch16
# @IDE: PyCharm
# @Author: ZhuangYuZhou
# @E-mail: 605540375@qq.com
# @Time: 2021/12/27
# @Desc: 
# # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # #

import torch
import torch.nn as nn
import numpy as np

class DiceCELoss(nn.Module):
    """Dice loss tailored to Brats need.
    """

    def __init__(self, lambda_region=0.8, lambda_dis=0.2, do_sigmoid=True):
        super(DiceCELoss, self).__init__()
        self.do_sigmoid = do_sigmoid
        self.labels = ["ET", "TC", "WT"]
        self.device = "cpu"
        self.lambda_region = lambda_region
        self.lambda_dis = lambda_dis

    def binary_dice(self, inputs, targets, label_index, metric_mode=False):
        # origin: e = 1
        smooth = 1e-6
        if self.do_sigmoid:
            inputs = torch.sigmoid(inputs)

        if metric_mode:
            inputs = inputs > 0.5
            if targets.sum() == 0:
                print(f"No {self.labels[label_index]} for this patient")
                if inputs.sum() == 0:
                    return torch.tensor(1., device="cuda")
                else:
                    return torch.tensor(0., device="cuda")

        intersection = DiceCELoss.compute_intersection(inputs, targets)

        if metric_mode:
            dice = (2 * intersection) / ((inputs.sum() + targets.sum()) * 1.0)
        else:
            dice = (2 * intersection + smooth) / (inputs.pow(2).sum() + targets.pow(2).sum() + smooth)

        if metric_mode:
            return dice

        return 1 - dice

    def bce_loss(self, inputs, targets):
        inputs = torch.sigmoid(inputs)
        targets = targets.float()

        bce_criterion = nn.BCEWithLogitsLoss()
        bce_loss = bce_criterion(input=inputs, target=targets)

        return bce_loss


    @staticmethod
    def compute_intersection(inputs, targets):
        intersection = torch.sum(inputs * targets)
        return intersection

    def forward(self, inputs, target):
        dice = 0
        for i in range(target.size(1)):
            dice = dice + self.binary_dice(inputs[:, i, ...], target[:, i, ...], i)
        final_dice = dice / target.size(1)
        dice_loss = final_dice
        bce_loss = self.bce_loss(inputs=inputs, targets=target)
        combo_loss = 0.8*dice_loss + 0.2*bce_loss
        return combo_loss

    def metric(self, inputs, target):
        dices = []
        for j in range(target.size(0)):
            dice = []
            for i in range(target.size(1)):
                dice.append(self.binary_dice(inputs[j, i], target[j, i], i, True))
            dices.append(dice)
        return dices

