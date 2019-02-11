#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Wed Jan 30 09:13:18 2019

@author: tony
"""

import torch
import torch.nn as nn
import torch.nn.functional as F
from networks.attention import MultiHeadAttention


def res_(chanIn, chanOut, ks = 3, stride=1, padding=1):
    return nn.Sequential(
        nn.Conv3d(chanIn, chanOut, ks, stride, padding=padding),
        nn.BatchNorm3d(chanOut,momentum=.997),
        nn.ReLU6(inplace=True),
        nn.Conv3d(chanOut, chanOut, ks, stride=1, padding=1)        
    )
    
def projection(chanIn, chanOut, ks=1, stride=1):
    return nn.Sequential(
        nn.Conv3d(chanIn, chanOut, ks, stride)
    )
        
def bn_relu(chanIn):
    return nn.Sequential(
            nn.BatchNorm3d(chanIn, momentum=.997),
            nn.ReLU6(inplace=True)
    )

class Block_B(nn.Module): 
    def __init__(self, chanIn, chanOut,stride,padding):
        super().__init__()
        self.stride = stride        
        self.top_bn_relu = bn_relu(chanIn)
        self.projection = projection(chanIn, chanOut,ks=1,stride=stride)
        self.residual = res_(chanIn, chanOut,ks=3,stride=stride, padding=padding)
        
    def forward(self, x):
        x_top = self.top_bn_relu(x)
        shortcut = self.projection(x_top)
        res_ = self.residual(x_top)
        return torch.add(res_,shortcut)
        
class Block_C(nn.Module):
    '''
    input = input from encoder
    output: input + MultiHeadAttention
    '''
    def __init__(self, chanIn, chanOut, heads):
        super().__init__()
        self.bn_relu = bn_relu(chanIn)
        self.att = MultiHeadAttention(chanIn, chanOut, chanOut, chanOut, heads)
        
    def forward(self, x):
        x = self.bn_relu(x)
        att = self.att(x)
        return torch.add(x, att)
            

def decode(chanIn, chanOut, ks=3):
    return nn.Sequential()
                                 
    
class Block_D(nn.Module):
    '''
    Deconding block with attention
    '''
    def __init__(self, chanIn, chanOut, heads, skip):
        super().__init__()
        self.bn_relu = bn_relu(chanIn)
        self.deconv1 = nn.ConvTranspose2d(chanIn, chanOut, 3,2, padding=1, output_padding=1)
        self.att = MultiHeadAttention(chanIn, chanOut, chanOut,chanOut, heads, layer_type='UP')
        self.bn_relu2 = bn_relu(chanOut)
        self.res = res_(chanOut, chanOut)
        
        
    def forward(self, x, skip):
        top_x = self.bn_relu(x)
        shortcut = self.deconv1(top_x)
        att = self.att(top_x)
#        print(att.shape, shortcut.shape)
        att_shortcut = torch.add(shortcut, att)
#        print(att_shortcut.shape, skip.shape)
        skip_con = torch.add(att_shortcut, skip)
        bn_res = self.bn_relu2(skip_con)
        out = self.res(bn_res)
        return out
    
class Block_E(nn.Module):
    '''
    Deconding block without attention    
    '''
    def __init__(self, chanIn,chanOut,stride,outpad,skip):
        super().__init__()
        self.deconv1 = nn.ConvTranspose3d(chanIn, chanOut, 3,stride=stride, padding=1, output_padding=outpad)
        self.bn_relu = bn_relu(chanOut)
        self.res_ = res_(chanOut,chanOut)
    
    def forward(self, x, skip):
        deconv = self.deconv1(x)
#        print('deconv e', deconv.shape)
        skip_con = torch.add(deconv,skip)
        bn_relu = self.bn_relu(skip_con)
        res_ = self.res_(bn_relu)
        out = torch.add(skip_con,res_)
        return out

def out_block(chanIn):
    return nn.Sequential(
            bn_relu(chanIn),
            nn.Dropout3d(.5),
            nn.Conv3d(chanIn,1,1,1)         
    )
        

class Model_B_E(nn.Module):
    def __init__(self, input_size):
        super().__init__()
        self.chn = input_size
        self.input_conv1 = nn.Conv3d(1,self.chn,kernel_size=3,stride=1,padding=1)
        self.block_1 = Block_B(self.chn, self.chn, stride=1, padding=1)
        self.block_2 = Block_B(self.chn, self.chn*2, stride=2, padding=1)
        self.block_3 = Block_B(self.chn*2, self.chn*4, stride=(2,2,1), padding=1)
        self.bottom = nn.Sequential(
                        bn_relu(self.chn*4),
                        nn.Conv3d(self.chn*4,self.chn*4,kernel_size=3,stride=1,padding=1))        
        self.block4 = Block_E(self.chn*4, self.chn*2,stride=(2,2,1),outpad=(1,1,0), skip=None)
        self.block5 = Block_E(self.chn*2, self.chn, stride=(2,2,2),outpad=(1,1,1),skip=None)
        self.out = out_block(self.chn)
        
        
    def forward(self, x):
        input_ = self.input_conv1(x)
        print('input', input_.shape)
        block1 = self.block_1(input_)
        print('block1', block1.shape)
        block2 = self.block_2(block1)
        print('block2', block2.shape)
        block3 = self.block_3(block2)
        print('block3', block3.shape)
        bottom = self.bottom(block3)
        print('bott', bottom.shape)
        
        block4 = self.block4(bottom, block2)
        print('block4: ',block4.shape)
        block5 = self.block5(block4,block1)
        x = self.out(block5)  
        print('out last shape: ', x.shape)
#        x = x.view(x.shape[0], x.shape[1],-1)
#        x =  F.log_softmax(x,dim=1)
        return x
        
class Model_B_C_E(nn.Module):
    def __init__(self, input_size, heads):
        super().__init__()
        self.chn = input_size
        self.input_conv1 = nn.Conv3d(1,self.chn,kernel_size=3,stride=1,padding=1)
        self.block_1 = Block_B(self.chn, self.chn, stride=1, padding=1)
        self.block_2 = Block_B(self.chn, self.chn*2, stride=2, padding=1)
        self.block_3 = Block_B(self.chn*2, self.chn*4, stride=(2,2,1), padding=1)
        
        self.bottom = Block_C(self.chn*4, self.chn*4, heads)
        
        self.block4 = Block_E(self.chn*4, self.chn*2,stride=(2,2,1),outpad=(1,1,0), skip=None)
        self.block5 = Block_E(self.chn*2, self.chn, stride=(2,2,2),outpad=(1,1,1),skip=None)
        self.out = out_block(self.chn)
        
        
    def forward(self, x):
        input_ = self.input_conv1(x)
#        print('input', input_.shape)
        block1 = self.block_1(input_)
#        print('block1', block1.shape)
        block2 = self.block_2(block1)
#        print('block2', block2.shape)
        block3 = self.block_3(block2)
#        print('block3', block3.shape)
        bottom = self.bottom(block3)
#        print('bott', bottom.shape)
        
        block4 = self.block4(bottom, block2)
#        print('block4: ',block4.shape)
        block5 = self.block5(block4,block1)
        x = self.out(block5)        
#        x = x.view(x.shape[0], x.shape[1],-1)
#        x =  F.log_softmax(x,dim=1)
#        print('out:...',x.shape)
        return x
    
class Model_B_C_D_E(nn.Module):
    def __init__(self, input_size, heads):
        super().__init__()
        self.chn = input_size
        self.input_conv1 = nn.Conv2d(1,self.chn,kernel_size=3,stride=1,padding=1)
        self.block_1 = Block_B(self.chn, self.chn, stride=1, padding=1)
        self.block_2 = Block_B(self.chn, self.chn*2, stride=2, padding=1)
        self.block_3 = Block_B(self.chn*2, self.chn*4, stride=2, padding=1)
        
        self.bottom = Block_C(self.chn*4, self.chn*4, heads)
        
        self.block4 = Block_D(self.chn*4, self.chn*2, heads,skip=None)#16
        self.block5 = Block_E(self.chn*2, self.chn, skip=None)
        self.out = out_block(self.chn)
        
        
    def forward(self, x):
        input_ = self.input_conv1(x)
#        print('input', input_.shape)
        block1 = self.block_1(input_)
#        print('block1', block1.shape)
        block2 = self.block_2(block1)
#        print('block2', block2.shape)
        block3 = self.block_3(block2)
#        print('block3', block3.shape)
        bottom = self.bottom(block3)
#        print('bott', bottom.shape)
        
        block4 = self.block4(bottom, block2)
#        print('block4: ',block4.shape)
        block5 = self.block5(block4,block1)
        x = self.out(block5)        
        x = x.view(x.shape[0], x.shape[1],-1)
        x =  F.log_softmax(x,dim=1)
        return x
        
            
class Att_Net(nn.Module):
    def __init__(self, input_size):
        self.chn = input_size
        super().__init__()
        self.input_conv1 = nn.Conv2d(1,self.chn,3,1,1)
        self.block1_res1 = res_(self.chn, self.chn)        
        self.block1 = bn_relu(self.chn, self.chn,ks=3,stride=1,padding=1)
        
        self.block2_res2 = res_(self.chn, self.chn*2,ks=1,stride=2)        
        self.block2 = bn_relu(self.chn, self.chn*2,ks=3,stride=2)
        
        self.block3_res3 = res_(self.chn*2, self.chn*4,ks=1,stride=2)        
        self.block3 = bn_relu(self.chn*2, self.chn*4,ks=3,stride=2)
#        
#        
#        self.block2 = bn_relu(self.chn, self.chn*2,3,2) #down
#        self.block2_res = nn.Conv2d(self.chn, self.chn*2,1,2, padding=0)
#        
#        self.block3 = bn_relu(self.chn*2, self.chn*4, 3,2)
#        self.block3_res = nn.Conv2d(self.chn*2, self.chn*4,1,2, padding=0)
        self.bn = nn.BatchNorm2d(self.chn*4,momentum=.997)
        self.mid = MultiHeadAttention(self.chn*4,self.chn*4,32,32,8)
        
        
        
        self.up2 = nn.ConvTranspose2d(self.chn*4, self.chn*2, 3,2, padding=1, output_padding=1)
        self.up2_1 = bn_relu(self.chn*2,self.chn*2, 3,1,1)

        self.up1 = nn.ConvTranspose2d(self.chn*2, self.chn, 3,2, padding=1, output_padding=1)
        self.up1_1 = bn_relu(self.chn,self.chn,3,1,1)
#         self.up1m = MultiHeadAttention_(32,32,12,12,4)
#         self.bn3 = nn.BatchNorm2d(32)     
        
        self.out_bn = nn.BatchNorm2d(self.chn,momentum=.997)
        self.drop = nn.Dropout2d(.5)
        self.out = bn_relu(self.chn, self.chn,3,1,1)
        self.out_ = nn.Conv2d(self.chn,2,1,1)      
        
        
    def forward(self, x):
#        print('input shape ', x.shape)
        x = self.input_conv1(x)
        x_top = self.block1_res1(x)
        x = self.block1(x_top)
        res1 = torch.add(x_top,x) #res1 32
#        print('res1 shape: ', res1.shape)
        
        x_top = self.block2_res2(res1)
#        print('x top shape: ', x_top.shape)
        x = self.block2(res1)
#        print('x  shape: ', x.shape)
        res2 = torch.add(x_top, x)#res2 64
#        print('res2 shape: ', x_res2.shape)
        
        x_top = self.block3_res3(res2)
#        print('x top shape: ', x_top.shape)
        x = self.block3(res2)
#        print('x top shape: ', x.shape)
        res3 = torch.add(x_top, x)#res2 128
#        print('res2 shape: ', x_res2.shape)
        
#       res3 shape  torch.Size([32, 128, 12, 12])
        x = self.bn(res3)
        x = self.mid(res3)
        
#        print('out mid: ', x.shape)
        
        x = self.up2(x)    
#        print('out transpose ', x.shape)
        x = torch.add(x, res2)
#        print('x after add ', x.shape)
        x_l = self.up2_1(x)
        x_out = torch.add(x_l,x)
        
        
        
        x = self.up1(x_out)
        x = torch.add(x, res1)
        x_l = self.up1_1(x)
        x_out = torch.add(x_l,x)
        
        
        x = self.out_bn(x_out)
        x = self.drop(x)
        x = self.out(x)
        x = self.out_(x)
#         print('before perm ', x.shape)
        x = x.view(x.shape[0], x.shape[1],-1)
#         x = x.permute(0,2,1)
        
#         print('out : ', x.shape)
        x =  F.log_softmax(x,dim=1)
#         print('outshape', x.shape)
        return x
    
