#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Sun Jan 13 22:56:10 2019

@author: tony
"""
import torch
import torch.utils.data
from imgaug import augmenters as iaa
import random
import numpy as np

class Neuron_dataset(torch.utils.data.Dataset):

    def __init__(self,images, train=True, masks=None, transform=True):
        
        self.train = train
        self.images = images
        self.transform = transform
        #default dimensions
        self.dim_x = 64
        self.dim_y = 64
        self.dim_z = 6
        if self.train:
            self.masks = masks
            
    def transform_(self, image, mask):
#         print('transform')
        if random.random() > .5 :
            image = np.fliplr(image).copy()
            mask = np.fliplr(mask).copy()
                             
        if random.random() > .5:
            image = np.flipud(image).copy()
            mask = np.flipud(mask).copy()
                             
        if random.random() > .5:
            image = np.rot90(image).copy()
            mask = np.rot90(mask).copy()

         
        image = image.reshape(1,self.dim_x, self.dim_y,self.dim_z)
        mask = mask.reshape(1,self.dim_x, self.dim_y,self.dim_z)
#         
#         https://imgaug.readthedocs.io/en/latest/source/augmenters.html
#         if random.random() > .5:
#             aug = iaa.GaussianBlur(sigma=(1.0,2.0))
#             image = aug.augment_images(image)
            
#         if random.random() > .5:
#             aug = iaa.PiecewiseAffine(scale=(.01,.06))
#             image = aug.augment_images(image)
            
            
        return image, mask

    def __len__(self):
        return len(self.images)

    def __getitem__(self, idx):
        image = self.images[idx]
        
        if self.train==True:
#            print('train is true')
            if self.transform==True:
                mask = self.masks[idx]
                return self.transform_(image, mask)
            else:
                mask = self.masks[idx]
                image = image.reshape(1,self.dim_x, self.dim_y,self.dim_z)
                mask = mask.reshape(1,self.dim_x, self.dim_x, self.dim_y,self.dim_z)                
                return image, mask
        
#        print('train is false')
        image = image.reshape(1,self.dim_x, self.dim_y,self.dim_z)
        return image
    