#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Mon Jan 21 15:24:37 2019

@author: tony
"""
import torch
import torch.nn as nn


def conv_layer(chanIn, chanOut, ks = 3, stride=1):
    return nn.Sequential(
        nn.Conv3d(chanIn, chanOut, ks, stride, padding=1),
        nn.ReLU(),
        nn.BatchNorm3d(chanOut)
        )

class Conv_block(nn.Module):
    def __init__(self, chanIn, chanOut, pool = False):
        super().__init__()
        
        self.conv1 = conv_layer(chanIn, chanOut)
        self.conv2 = conv_layer(chanOut, chanOut)
        
    def forward(self, x) : 
        x = self.conv1(x)
        x = self.conv2(x)
        return x
        
class Mid_block(nn.Module):
    def __init__(self, chanIn, chanOut, ks=3, stride=1):
        super().__init__()        
        self.conv1 = nn.Conv3d(chanIn, chanOut, ks, padding=1)
        self.conv2 = nn.Conv3d(chanOut, chanOut, ks, padding=1)
        
    def forward(self, x) : 
        x = self.conv1(x)
        x = self.conv2(x)
        return x