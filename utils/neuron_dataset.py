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
        """
        Args:
            text_file(string): path to text file
            root_dir(string): directory with all train images
        """
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

        
        randy = random.randint(0,7)
            
        if randy == 0:
            pass #nothing, just a normal image
        
        #flip left to right only
        if randy == 1:
            image = np.fliplr(image).copy()
            mask = np.fliplr(mask).copy()

        #flip up or down only
        if randy ==2:
            image = np.flipud(image).copy()
            mask = np.flipud(mask).copy()

        #rotate 90 degrees
        if randy ==3:
            image = np.rot90(image).copy()
            mask = np.rot90(mask).copy()
            
        #flip lr and up
        if randy ==4:
            image = np.fliplr(image).copy()
            mask = np.fliplr(mask).copy()
            image = np.flipud(image).copy()
            mask = np.flipud(mask).copy()
            
        if randy ==5:
            image = np.fliplr(image).copy()
            mask = np.fliplr(mask).copy()
            image = np.rot90(image).copy()
            mask = np.rot90(mask).copy()
        
        if randy ==6:
            image = np.flipud(image).copy()
            mask = np.flipud(mask).copy()
            image = np.rot90(image).copy()
            mask = np.rot90(mask).copy()
            
            
#
        #need to reshape here before using iaa augs
        image = image.reshape(1,self.dim_x, self.dim_y,self.dim_z)
        mask = mask.reshape(1,self.dim_x, self.dim_y,self.dim_z)
#         https://imgaug.readthedocs.io/en/latest/source/augmenters.html
        
        if self.transform:
            pass
#             if random.random() > .9:
#                 aug = iaa.GaussianBlur(sigma=(1.0,2.0))
#                 image = aug.augment_images(image)

#             if random.random() > .9:
#                 aug = iaa.PiecewiseAffine(scale=(.01,.06))
#                 image = aug.augment_images(image)
            
            
        return image, mask

    def __len__(self):
        return len(self.images)

    def __getitem__(self, idx):
        image = self.images[idx]
        
        if self.train:
            mask = self.masks[idx]
            return self.transform_(image, mask)
        
        image = image.reshape(1,self.dim_x, self.dim_y,self.dim_z)
        return image
    
    