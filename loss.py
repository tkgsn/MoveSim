# coding: utf-8
import pdb
import torch
import numpy as np
import torch.nn as nn
from torch.autograd import Variable

from my_utils import get_gps


class GANLoss(nn.Module):
    """Reward-Refined NLLLoss Function for adversial training of Gnerator"""

    def __init__(self):
        super(GANLoss, self).__init__()

    def forward(self, probs, targets, reward, ploss=False):
        # get the device of prob
        device = probs.device

        probs = probs.reshape(-1, probs.shape[-1])
        one_hot = torch.zeros_like(probs)
        one_hot.scatter_(1, targets.data.view((-1, 1)), 1)
        one_hot = one_hot.type(torch.bool).to(device)
        # one_hot = Variable(one_hot)
        # print(probs.shape, one_hot.shape)
        loss = 1-torch.masked_select(probs, one_hot)
        loss = loss * reward
        loss = torch.mean(loss)
        return loss