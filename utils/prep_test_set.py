#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Mon Jan 21 15:55:57 2019

@author: tony

3d dataset for inference

"""

import numpy as np
def make_test_set(img, chunks):
    
    '''
    chunks should be the multiplier of 128x128xchunks
    where chunks is 6 layers * chunks, thus this will determine
    how many loops in the z direction.  For example chunks=5
    will be 5x6 = 30
    
    Since this takes up too much memory, chunk it into pieces.
    128*128 are the number of samples every 8 pix *5 for 6*5 for 30 layers.
    
    We are starting at pix 64 and taking 30 left and 34 right, same for up and down
    in 6 layer incs. in the z direction.
    
    '''
    
    #size of dataset
    test_l =128*128*chunks #256*256#512*512# 1638400# 256*256#5184 *
    
    dim_z = 6
    dim_x = 64
    dim_y = 64
    X_test = np.zeros((test_l, dim_x, dim_y, dim_z), dtype=np.float32)

    count =0
    for k in range(0,6*chunks,6):
        for i in range(64, 1088,8):
            for j in range( 64,1088,8):    
                X_test[count] = img[i-30:i+34, j-30:j+34,k:k+6]
                count+=1
    print('dataset size: ', count)
    
    X_test /=255
    return X_test
    
    #print(count)