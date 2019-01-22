##!/usr/bin/env python3
## -*- coding: utf-8 -*-
#"""
#Created on Sun Jan 13 22:50:22 2019
#
#@author: tony
#"""
import torch
import torch.nn as nn
from networks.unet_utils import Conv_block, Mid_block


  

class Unet(nn.Module):
    def __init__(self):
        super(Unet, self).__init__()
        
        self.down1 = Conv_block(1,16)
        self.down2 = Conv_block(16, 32)
        self.down3 = Conv_block(32,64)
        self.down4 = Conv_block(64,128)
        
        self.mid = Mid_block(128,256)
        
        self.upt3 = nn.ConvTranspose3d(256,128,(2,2,1),(2,2,1))
        self.up3 =Conv_block(256,128)
        self.upt2 = nn.ConvTranspose3d(128, 64,(2,2,1),(2,2,1))
        self.up2 = Conv_block(128,64)
        self.upt1 = nn.ConvTranspose3d(64, 32,(2,2,1),(2,2,1))
        self.up1 = Conv_block(64,32)
        self.upt0 = nn.ConvTranspose3d(32, 16,(2,2,1),(2,2,1))
        self.up0 = Conv_block(32,16)
        self.last = nn.Conv3d(16,1,1) #output channels 1 or 2
        
        self.maxPool = nn.MaxPool3d((2,2,1))
        self.dropout = nn.Dropout3d(p=.50, inplace=True)
        
    
    def forward(self, x):
#         print('input shape', x.shape)
        x1 = self.down1(x)
        p1 = self.maxPool(x1)
        p1 = self.dropout(p1)
#         print('p1 shape', p1.shape)
        x2 = self.down2(p1)
#         print('x2 shape', x2.shape)
        p2 = self.maxPool(x2)
        p2 = self.dropout(p2)
        
        x3 = self.down3(p2)
        p3 = self.maxPool(x3)
        p3 = self.dropout(p3)
        
        x4 = self.down4(p3)
        p4 = self.maxPool(x4)
        p4 = self.dropout(p4)
#         print('looking', p4.shape)
        xmid = self.mid(p4)
        
        xu3 = self.upt3(xmid)
#         print('xu3', xu3.shape)
#         print('x4', x4.shape)
        cat3 = torch.cat([x4, xu3],1) #x4,xu3
        xu3 = self.up3(cat3)
        
        xu2 = self.upt2(xu3)
        cat2 = torch.cat([x3, xu2],1)
        xu2 = self.up2(cat2)
        
        xu1 = self.upt1(xu2)
        cat1 = torch.cat([x2,xu1],1)
        xu1 = self.up1(cat1)
        
        xu0 = self.upt0(xu1)
        cat0 = torch.cat([x1,xu0],1)
        xu0 = self.up0(cat0)
        out = self.last(xu0)
#         print('output shape ',out.shape)
        return (out)


