#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Wed Jan 23 08:19:21 2019

@author: tony
"""

import torch
import torch.nn as nn


def conv_layer(chanIn, chanOut, ks = (3,3,1), stride=1):
    return nn.Sequential(
        nn.Conv3d(chanIn, chanOut, ks, stride, padding=(1,1,0)),
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
    
class Res_block(nn.Module):
    def __init__(self, chanIn, chanOut, pool=False):
        super(Res_block, self).__init__()
              
        self.conv_3x3x1 = nn.Conv3d(chanIn, chanOut,(3,3,1),1,padding=(1,1,0))
        self.bn1 = nn.BatchNorm3d(chanOut)
        self.conv_3x3x3_1 = nn.Conv3d(chanOut, chanOut,(3,3,3),1,padding=1)
        self.bn2 = nn.BatchNorm3d(chanOut)
        self.conv_3x3x3_2 = nn.Conv3d(chanOut, chanOut,(3,3,3),1,padding=1)
        self.bn3 = nn.BatchNorm3d(chanOut)
        self.activation = nn.ReLU()
        
    def forward(self, x):
#         print('shape init res: ', x.shape)
        x = self.conv_3x3x1(x)
        x = self.bn1(x)
        x = self.activation(x)
#         print('shape after 331 res: ', x.shape)
        residual = x
        
        x = self.conv_3x3x3_1(x)
        x = self.bn2(x)
        x = self.activation(x)
        
        x = self.conv_3x3x3_2(x)
        x = self.bn3(x)
        x = self.activation(x)
#         print('shape before add ', x.shape)
        x += residual
#         print('shape after add ', x.shape)
        return x

def conv_1x(chanIn, chanOut):
    return nn.Conv3d(chanIn, chanOut,(1,1,1))
        
class Mid_block(nn.Module):
    def __init__(self, chanIn, chanOut, ks=3, stride=1):
        super().__init__()        
        self.conv1 = nn.Conv3d(chanIn, chanOut, ks, padding=1)
        self.conv2 = nn.Conv3d(chanOut, chanOut, ks, padding=1)
        
    def forward(self, x) : 
        x = self.conv1(x)
        x = self.conv2(x)
        return x
    
class Unet_res(nn.Module):
    def __init__(self):
        super(Unet_res, self).__init__()
        
        self.down1 = Conv_block(1,16)
        self.down2 = Res_block(16, 16)
        self.filt_up1 = conv_1x(16,32)
        self.down3 = Res_block(32,32)
        self.filt_up2 = conv_1x(32, 64)
        self.down4 = Res_block(64,64)
        self.filt_up3 = conv_1x(64, 128)
        
        self.mid = Res_block(128,128)
        
        self.upt3 = nn.ConvTranspose3d(128,64,(2,2,1),(2,2,1))
        self.up3 =Res_block(64,64)#(256,128)
        self.upt2 = nn.ConvTranspose3d(64, 32,(2,2,1),(2,2,1))
        self.up2 = Res_block(32,32)
        self.upt1 = nn.ConvTranspose3d(32,16,(2,2,1),(2,2,1))
        self.up1 = Res_block(16,16)
        self.upt0 = nn.ConvTranspose3d(16, 16,(2,2,1),(2,2,1))
        self.up0 = Res_block(16,16)
        self.last = nn.Conv3d(16,1,1) #output channels 1 or 2
        
        self.maxPool = nn.MaxPool3d((2,2,1))
#         self.maxPool2 = nn.MaxPool3d((2,2,2))
        self.dropout = nn.Dropout3d(p=.50, inplace=True)
        
    
    def forward(self, x):
#         print('input shape', x.shape)
        #save _a for res addition
        x1_a = self.down1(x)
#         print('input shape', x1_a.shape)
        p1 = self.maxPool(x1_a)
#         print('input shape', p1.shape)
        p1 = self.dropout(p1)
#         print('p1 shape', p1.shape)
        x2_a = self.down2(p1)
#         print('after first res', x2_a.shape)
        x2 = self.filt_up1(x2_a)
#         print('up1 shape', x2.shape)
        p2 = self.maxPool(x2)
        p2 = self.dropout(p2)
        
        
        x3_a = self.down3(p2)
        x3 = self.filt_up2(x3_a)
        p3 = self.maxPool(x3)
        p3 = self.dropout(p3)
        
        x4_a = self.down4(p3)
        x4 = self.filt_up3(x4_a)
        p4 = self.maxPool(x4)
        p4 = self.dropout(p4)
#         print('looking p4', p4.shape)
        xmid = self.mid(p4)
        
        xu3 = self.upt3(xmid)
#         print('x4_a: ', x4_a.shape)
#         print('xu3 :', xu3.shape)
        add3 = torch.add(x4_a,xu3)#([x4, xu3],1) #x4,xu3
#         print('cat3', cat3.shape)
        xu3 = self.up3(add3)
        
        xu2 = self.upt2(xu3)
#         print('x3', x3.shape)
#         print('xu2', xu2.shape)
        add2 = torch.add(x3_a, xu2)
        xu2 = self.up2(add2)
        
        xu1 = self.upt1(xu2)
        add1 = torch.add(x2_a,xu1)
        xu1 = self.up1(add1)
        
        xu0 = self.upt0(xu1)
        add0 = torch.add(x1_a,xu0)
        xu0 = self.up0(add0)
        out = self.last(xu0)
#         print('output shape ',out.shape)
        return (out)