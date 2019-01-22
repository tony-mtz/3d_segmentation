#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Mon Jan 21 15:31:40 2019

@author: tony
"""

import matplotlib.pyplot as plt
import numpy as np

def image_gray(img, sz=10, color='gray'):
    plt.figure()
    plt.subplots(figsize=(sz,sz))
    plt.imshow(img, cmap=color)
    plt.show()
    
'''
preprocess=True takes an array img[0][:,:,0]
Is just to peek at the top z dimension results    
'''
def image_grid(img, preprocess=True, grid_x=9, grid_y=8,fig_size=(18,18)):
   
    fig, axes = plt.subplots(grid_x,grid_y, figsize=fig_size,
                         subplot_kw={'xticks': [], 'yticks': []})
    fig.subplots_adjust(hspace=0.05, wspace=0.05)

    for ax, i in zip(axes.flat, range(len(img))):
        
        if preprocess:
            ax.imshow(img[i][:,:,0], cmap='gray')
        else:
            ax.imshow(img[i], cmap='gray')
    plt.show()
    
    
#8x8
def rebuild(img):
    res_ = np.zeros((1024,1024,24))
    count =0
    #z to 24 by 6
    for k in range(0,24,6):
        #j and i 128 for stride of 8
        for j in range(128):
    
            for i in range(128):
            #     for j in range(256,4):
                res_[j*8:j*8+8,i*8:8*i+8,k:k+6]= img[count][28:36,28:36,:6]
                count +=1
    print(count)
    return res_