#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Tue Aug  3 14:03:05 2021

@author: Amoko
"""

from config import MyConfig
cfg = MyConfig()

import os
import csv
import random
import numpy as np
from PIL import Image
import torch
from torch.utils.data import Dataset, DataLoader
from torchvision import transforms
import torchvision.transforms.functional as F
from torchvision.utils import save_image
import numbers

class MyPad(object):
    def __init__(self, fill=0, padding_mode='constant'):
        assert isinstance(fill, (numbers.Number, str, tuple))
        assert padding_mode in ['constant', 'edge', 'reflect', 'symmetric']

        self.fill = fill
        self.padding_mode = padding_mode
        
    def __call__(self, img):
        padding = self.get_padding(img)
        return F.pad(img, padding, self.fill, self.padding_mode)
    
    def __repr__(self):
        return self.__class__.__name__ + '(padding={0}, fill={1}, padding_mode={2})'.\
            format('', self.fill, self.padding_mode)
    
    def get_padding(self, img):
        w, h = img.size
        max_wh = np.max([w, h])
        h_padding = (max_wh - w) / 2
        v_padding = (max_wh - h) / 2
        l_pad = h_padding if h_padding % 1 == 0 else h_padding+0.5
        t_pad = v_padding if v_padding % 1 == 0 else v_padding+0.5
        r_pad = h_padding if h_padding % 1 == 0 else h_padding-0.5
        b_pad = v_padding if v_padding % 1 == 0 else v_padding-0.5
        padding = (int(l_pad), int(t_pad), int(r_pad), int(b_pad))
        return padding

class AngleDataset(Dataset):
    ''' Image Angle dataset. '''

    def __init__(self, csv_path, mode, transform=None, enhance=False, cut_corner=False):
        self.lines = self.read_csv(csv_path)
        split = int(len(self.lines)*0.8)
        if mode == 'train':
            self.lines = self.lines[:split]
        else:
            self.lines = self.lines[split:]
        self.sample_num = len(self.lines)
        
        self.transform = transform
        self.enhance = enhance
        self.cut_corner = cut_corner

    def read_csv(self, csv_name):
        with open(csv_name) as f:
            csv_reader = csv.reader(f, delimiter=',')
            lines = list(csv_reader)
        for i in range(len(lines)):
            #print(lines[i])
            lines[i][1] = float(lines[i][1])
            lines[i][2] = float(lines[i][2])
            lines[i][3] = float(lines[i][3])
        random.shuffle(lines)
        return lines
    
    def get_legal_plus(self, angle):
        if cfg.sample_type == 'continuous':
            start, end = cfg.sample_continuous
            plus = random.uniform(start, end)
        else:
            plus = random.choice(cfg.sample_discrete)
            
        angle_plus = angle + plus
        if angle_plus < -180 or angle_plus > 180:
            plus = 0
        return plus
    
    def get_rotation_img(self, img, plus):
        p = MyPad()
        img = p(img)
        img_rotation = img.rotate(-plus)
        return img_rotation
    
    def angle2vector(self, angle):
        radian = np.radians(angle)
        cos = np.cos(radian)
        sin = np.sin(radian)
        return [sin, cos]
        
    def cut_upper_right(self, img):
        w, h = img.size
        img_np = np.array(img)
        #img_np.setflags(write=1)
        w_start, w_end = w//2, w
        #w_start, w_end = 0, w
        for r in range(0, h//2):
            for c in range(w_start, w_end):
                img_np[r][c] = [0,0,0]
            w_start += 1
        
        img2 = Image.fromarray(img_np)
        return img2
    
    def cut_upper_left(self, img):
        w, h = img.size
        img_np = np.array(img)
        #img_np.setflags(write=1)
        w_start, w_end = 0, w//2
        #w_start, w_end = 0, w
        for r in range(0, h//2):
            for c in range(w_start, w_end):
                img_np[r][c] = [0,0,0]
            w_end -= 1
        
        img2 = Image.fromarray(img_np)
        return img2

    def __len__(self):
        return len(self.lines)
    
    def __getitem__(self, idx):
        if torch.is_tensor(idx):
            idx = idx.tolist()
        fn, angle, x, y = self.lines[idx]
        
        #print(fn, angle)

        img = Image.open(fn).convert('RGB')
        if self.cut_corner:
            n = random.choice(list(range(10)))
            if n in [1]:
                img = self.cut_upper_left(img)
            elif n in [2]:
                img = self.cut_upper_right(img)
            
        if self.enhance:
            plus = self.get_legal_plus(angle)
            img = self.get_rotation_img(img, plus)
            angle += plus
            x, y = self.angle2vector(angle)

        if self.transform:
            img = self.transform(img)
            
        #angle_norm = angle/cfg.degree_scale
        
        return img, angle, torch.FloatTensor([x, y])
    

def get_transfroms():
    mean, std = [0.485, 0.456, 0.406], [0.229, 0.224, 0.225]
    
    train_transforms = transforms.Compose([
        MyPad(),
        transforms.Resize(cfg.IMG_SIZE),
        transforms.RandomAffine(0,translate=(0.04, 0.04)),
        transforms.ToTensor(),
        transforms.Normalize(mean, std)
        ])
    val_transforms = transforms.Compose([
        MyPad(),
        transforms.Resize(cfg.IMG_SIZE),
        transforms.ToTensor(),
        transforms.Normalize(mean, std)
        ])
    return train_transforms, val_transforms
    

def get_dataset(csv_path, train_transforms, val_transforms):
    mode, enhance, cut_corner = 'train', True, True
    train_dataset = AngleDataset(csv_path, mode, train_transforms, enhance, cut_corner)
    mode, enhance, cut_corner  = 'val', False, False
    val_dataset = AngleDataset(csv_path, mode, val_transforms, enhance, cut_corner)
    return train_dataset, val_dataset


def get_dataloader():
    train_transforms, val_transforms = get_transfroms()
    train_dataset, val_dataset = get_dataset(cfg.csv_path, 
                                             train_transforms, val_transforms)
    
    train_loader = DataLoader(
        train_dataset, batch_size=cfg.BATCH_SIZE, shuffle=True, num_workers=8)
    val_loader = DataLoader(
        val_dataset, batch_size=cfg.BATCH_SIZE, shuffle=True, num_workers=8)
    return train_loader, val_loader


if __name__ == '__main__':
    cfg.degree_scale = 1
    
    train_transforms, val_transforms = get_transfroms()
    train_dataset, val_dataset = get_dataset(cfg.csv_path, 
                                             train_transforms, val_transforms)
    
    for idx, (img, angle, vector) in enumerate(train_dataset):
        path_dir = 'tmp_train'
        if not os.path.exists(path_dir):
            os.mkdir(path_dir)
        save_image(img, '{}/{}_{:.2f}.jpg'.format(path_dir, idx, angle))
