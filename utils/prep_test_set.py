#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Mon Jan 21 15:55:57 2019

@author: tony

3d dataset for inference

"""

import numpy as np
def make_test_set(img):
    
    #size of dataset
    test_l =256*256#512*512# 1638400# 256*256#5184 #144x1447
    
    dim_z = 6
    dim_x = 64
    dim_y = 64
    X_test = np.zeros((test_l, dim_x, dim_y, dim_z), dtype=np.float32)

    count =0
    for k in range(0,24,6):
        for i in range(64, 1088,8):
            for j in range( 64,1088,8):    
                X_test[count] = img[i-30:i+34, j-30:j+34,k:k+6]
                count+=1
    print('dataset size: ', count)
    return X_test
    
    #print(count)