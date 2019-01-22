#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Mon Jan 21 18:21:01 2019

@author: tony m
"""
import torch
from torch.autograd import Variable
import torch.nn.functional as F
from utils.display_utils import image_gray

'''
save_best and save_last are paths
'''
def train_loop(train_loader, val_loader, test_image, model, optimizer, scheduler, 
               criterion,save_best, save_last, epochs):    
    
    device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")    
        
    mean_train_losses = []
    mean_val_losses = []    
#    mean_train_acc = []
#    mean_val_acc = []
#    maxValacc = -99999
    minLoss = 99999
    
    
    for epoch in range(epochs):
        scheduler.step()
        print('EPOCH: ',epoch+1)
#        train_acc = []
#        val_acc = []
        
        running_loss = 0.0
        
        model.train()
        count = 0
        for images, labels in train_loader:
            
            images = Variable(images).to(device)
            labels = Variable(labels).to(device)
            
            outputs = model(images) 
            optimizer.zero_grad()
            loss = criterion(outputs, labels)            
            loss.backward()
            optimizer.step()        
            
            running_loss += loss.item()
            count +=1
        
        print('Training loss:.......', running_loss/count)
    #     print('Training accuracy:...', np.mean(train_acc))
        mean_train_losses.append(running_loss/count)
            
        model.eval()
        count = 0
        val_running_loss = 0.0
        for images, labels in val_loader:
            images = Variable(images).to(device)
            labels = Variable(labels).to(device)        
            
            outputs = model(images)
            loss = criterion(outputs, labels)
            
            val_running_loss += loss.item()
            count +=1
    
        mean_val_loss = val_running_loss/count
        print('Validation loss:.....', mean_val_loss)
        
    #     print('Training accuracy:...', np.mean(train_acc))
    #     print('Validation accuracy..', np.mean(val_acc))
        
        mean_val_losses.append(mean_val_loss)
        
    #     mean_train_acc.append(np.mean(train_acc))
        
    #     val_acc_ = np.mean(val_acc)
    #     mean_val_acc.append(val_acc_)
        
       
        if mean_val_loss < minLoss:
            torch.save(model.state_dict(), save_best )
            print(f'NEW BEST Loss: {mean_val_loss} ........old best:{minLoss}')
            minLoss = mean_val_loss
            print('')
            
    #     if val_acc_ > maxValacc:
    #         torch.save(model.state_dict(), 'res/cam_40/best_acc_norm_10x10.pth' )
    #         print(f'NEW BEST Acc: {val_acc_} ........old best:{maxValacc}')
    #         maxValacc = val_acc_
        if epoch==1999:
            torch.save(model.state_dict(), save_last )
            print(f'Last: {mean_val_loss}')
            
        if epoch%20==0:
            print('inside')
            images = Variable(test_image.cuda())
            outputs = model(images)
            image_gray(F.sigmoid(outputs)[0].squeeze()[:,:,0].data.cpu().numpy(),3)
            image_gray(test_image[0].squeeze()[:,:,0],3)
        
        print('')
    return mean_train_losses, mean_val_losses