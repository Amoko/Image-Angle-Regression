#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Tue Aug  3 10:06:18 2021

@author: Amoko
"""

from config import MyConfig
cfg = MyConfig()

import time
import numpy as np
import torch
import torch.nn as nn
import torch.optim as optim
from torchvision import models
from torchsummary import summary

from torch.cuda.amp import autocast
from dataloader_csv import get_dataloader


def get_angle_from_vector(v1):
    '''
    Parameters
    ----------
    v1 : list

    Returns
    -------
    angle : float
    
    Examples
    -------
    >>> get_angle_from_vector([1, 1])
    45
    >>> get_angle_from_vector([-1, 1])
    -45
    >>> get_angle_from_vector([1, -1])
    135
    >>> get_angle_from_vector([-1, -1])
    -135
    '''
    v0 = np.array([0, 1])
    v1 = np.array(v1)
    
    # 1 norm
    norm = np.linalg.norm(v0) * np.linalg.norm(v1)
    
    # 2 dot product
    cosine = np.dot(v0, v1) / norm
    radian = np.arccos(cosine)
    angle = np.rad2deg(radian)
    
    # 3 cross product
    rho = np.rad2deg(np.arcsin(np.cross(v0, v1) / norm))
    if rho>0:
        angle *= -1
    return angle

def get_angle_diff(angles_gt, vectors):
    angles_pred = [get_angle_from_vector(e) for e in vectors]
    diff = [abs(x1-x2) for x1,x2 in zip(angles_pred,angles_gt)]
    #print(angles_gt[:5])
    #print(angles_pred[:5])
    #print(diff[:5], '\n')
    
    
    for i in range(len(diff)):
        if diff[i] > 180:
            diff[i] = 360 - diff[i] 
    return sum(diff) / len(diff)        

def eval_inference_time(model):
    img = torch.rand((32, 3, 224, 224))
    img = img.to(DEVICE)
    
    start = time.time()
    N = 3000
    for i in range(N):
        with autocast(enabled=cfg.amp):
            output = model(img)
    end = time.time()
    avg_time = (end-start) / N
    print(avg_time)

def get_model_regression(DEVICE=torch.device('cpu')):
    model = models.mobilenet_v2(pretrained=True)
    
    #'''
    model.classifier = nn.Sequential(
        nn.Dropout(0.2),
        nn.Linear(model.last_channel, 2),
        #nn.Tanh(),
        )
    #'''
    
    model = model.to(DEVICE)
    return model
    

def train(model, opt, DEVICE, train_loader, epoch):
    model.train()
    train_loss = 0
    margin = 0
    interval = len(train_loader.dataset) // (5*cfg.BATCH_SIZE)
    for batch_idx, (inputs, angles, labels) in enumerate(train_loader):
        inputs, labels = inputs.to(DEVICE), labels.to(DEVICE)
        with autocast(enabled=cfg.amp):
            outputs = model(inputs)
        #outputs = outputs.squeeze()
        #print(outputs.tolist()[:4])
        
        cos_sim = torch.cosine_similarity(outputs, labels)
        loss = 1 - cos_sim
        loss = loss.mean()
        
        opt.zero_grad()
        loss.backward()
        opt.step()
        
        train_loss += loss.item()
        diff = get_angle_diff(angles.tolist(), outputs.tolist())
        margin += diff
        if batch_idx % interval == 0:
            print('Train Epoch: {} [{}/{} {:.0f}%]\t loss: {:.4f} margin: {:.1f}'\
                  .format(epoch, batch_idx*cfg.BATCH_SIZE, len(train_loader.dataset),
                100 * batch_idx/len(train_loader), loss.item(), diff))

            
    train_loss /= batch_idx
    margin /= batch_idx
    print('train_set: average_loss: {:.4f}, average_margin: {:.1f}'.format(
        train_loss, margin))
    return train_loss, margin


def validation(model, DEVICE, val_loader, epoch):
    pass
    model.eval()
    val_loss = 0
    margin = 0
    with torch.no_grad():
        for batch_idx, (inputs, angles, labels) in enumerate(val_loader):
            inputs, labels = inputs.to(DEVICE), labels.to(DEVICE)
            with autocast(enabled=cfg.amp):
                outputs = model(inputs)
            outputs = outputs.squeeze()

            cos_sim = torch.cosine_similarity(outputs, labels)
            loss = 1 - cos_sim
            loss = loss.mean()

            val_loss += loss.item()
            diff = get_angle_diff(angles.tolist(), outputs.tolist())
            margin += diff
            
    val_loss /= batch_idx
    margin /= batch_idx
    print('val_set: average_loss: {:.4f}, average_margin: {:.1f}'.format(
        val_loss, margin))
    return val_loss, margin
            
if __name__ == '__main__':
    
    print('Start.', time.ctime())
    print('dataset:', cfg.csv_path)
    print('checkpoints:', cfg.old_checkpoint)
    print('lr:', cfg.LR)
    print('BATCH_SIZE:', cfg.BATCH_SIZE)
    print('IMG_SIZE:', cfg.IMG_SIZE)
    print('amp:', cfg.amp)
    print('degree_scale:', cfg.degree_scale) 
    print('sample_type:', cfg.sample_type)
    print('sample_continuous :', cfg.sample_continuous)
    print('sample_discrete:', cfg.sample_discrete)
    
    DEVICE = torch.device('cuda:0')
    # 0 data
    train_loader, val_loader = get_dataloader()
    print('train samples:', len(train_loader.dataset))
    print('val samples:', len(val_loader.dataset))

    # 1 model
    model = get_model_regression(DEVICE)
    print(summary(model, (3,224,224)))
    #eval_inference_time(model)
    
    if cfg.old_checkpoint:
        model.load_state_dict(torch.load(cfg.old_checkpoint))
    
    '''
    # classification 1000
    # Total params: 3,504,872
    # inference time: batch32, res224, 9.7ms,(fp32) 6.6ms(amp)
    
    # regression
    # Total params: 2,225,153
    # inference time: batch32, res224, 9.5ms(fp32), 6.4ms(amp)
    '''
    
    # 2 loss, opt
    #criterion = nn.MSELoss()
    opt = optim.Adam(model.parameters(),lr=cfg.LR)


    # 3 train
    best_val_margin = 100
    best_epoch = 0
    for epoch in range(1, 10000+1):
        train_loss, train_margin = train(model, opt, DEVICE, train_loader, epoch)
        
        val_loss, val_margin = validation(model, DEVICE, val_loader, epoch)

        if val_margin < best_val_margin:
            best_val_margin = val_margin
            best_epoch = epoch
            # save model
            path = 'mobilenet_v2.{:.1f}.pth'.format(best_val_margin)
            torch.save(model.state_dict(), path)
            
        if epoch % 100 == 0:
            path = 'mobilenet_v2.epoch_{}_{:.1f}.pth'.format(epoch, val_margin)
            torch.save(model.state_dict(), path)

        print('best_val_margin: {:.1f} at epoch {}'.format(best_val_margin, best_epoch))
        print(time.ctime())
        print('-'*60)
        #break
        
    
    print(torch.cuda.get_device_name(0))
    print('Over.', time.ctime())
    
