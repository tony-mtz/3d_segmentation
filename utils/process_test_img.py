#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Sun Jan 13 23:04:48 2019

@author: tony
"""

from skimage.io import imread, imshow
import numpy as np
import cv2

class process_test_img():
    def __init__(self,path, mod=64):
        
        self.path = path
        self.mod = mod
        
        
    def mod_image(self):
        
        img = imread(self.path)
        img = np.rollaxis(img, 0, 3)
        
        #for 128 it was :50,90,50,90
        topBorderWidth = self.mod
        bottomBorderWidth =self.mod
        leftBorderWidth = self.mod
        rightBorderWidth= self.mod
        
        outputImage = cv2.copyMakeBorder(
                     img, 
                     topBorderWidth, 
                     bottomBorderWidth, 
                     leftBorderWidth, 
                     rightBorderWidth, 
                     cv2.BORDER_REFLECT             
                  )
        
        top = outputImage[:,:,0:1]
        bottom = outputImage[:,:,99:100]
        new = np.concatenate((top,outputImage),axis=2)
        new_tb = np.concatenate((new,bottom),axis=2)
        return new_tb
    
