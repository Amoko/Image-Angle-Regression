#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Thu Aug  5 21:04:20 2021

@author: Amoko
"""

import os
import csv
import shutil
import numbers
import numpy as np
from PIL import Image

import torch
import torch.nn as nn
from torchvision import models, transforms
import torchvision.transforms.functional as F

#from torchsummary import summary



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


class AngleInference():
    def __init__(self, model_path):
        self.DEVICE = torch.device('cuda:0')
        self.model = self.load_model(model_path)
        self.val_transforms = self.get_transforms()
        
    def get_model_regression(self):
        model = models.mobilenet_v2(pretrained=False)
        model.classifier = nn.Sequential(
            nn.Dropout(0.2),
            nn.Linear(model.last_channel, 2),
            #nn.Tanh(),
            )
        model = model.to(self.DEVICE)
        return model
    
    def get_transforms(self, IMG_SIZE=(224,224)):
        mean, std = [0.485, 0.456, 0.406], [0.229, 0.224, 0.225]
        val_transforms = transforms.Compose([
            MyPad(),
            transforms.Resize(IMG_SIZE),
            transforms.ToTensor(),
            transforms.Normalize(mean, std)
            ])
        return val_transforms
    
    def load_model(self, model_path):
        model = self.get_model_regression()
        model.load_state_dict(torch.load(model_path))
        model.eval()
        return model
    
    def load_img(self, fn):
        try:
            img = Image.open(fn).convert('RGB')
        except Exception as e:
            print(e)
            return None
        
        img_tensor = self.val_transforms(img)
        img_tensor = img_tensor.unsqueeze(0)
        img_tensor = img_tensor.to(self.DEVICE)
        return img_tensor
    
    def get_angle_from_vector(self, v1):
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
    
    def predict(self, fn):
        img = self.load_img(fn)
        if img is None:
            return None
        with torch.no_grad():
            vector = self.model(img)
            print(vector)
            vector = vector.squeeze()
        vector = vector.tolist()
        angle = self.get_angle_from_vector(vector)
        return angle

def get_zero_degree_img(fn, plus):
    img = Image.open(fn).convert('RGB')
    p = MyPad()
    img = p(img)
    img = img.resize((224,224))
    img_rotation = img.rotate(plus)
    return img_rotation

        
def test_4img(model):
    lines = [
        ['/home/Amoko/data/0805_angle/0_1_I105601417_1_8_0.png', -177.7251],
        ['/home/Amoko/data/0805_angle/0_2_I112633520_1_2_0.png', -96.5176],
        ['/home/Amoko/data/0805_angle/0_1_I105569557_4_2_0.png', 44.2717],
        ['/home/Amoko/data/0805_angle/0_2_I115334238_1_2_0.png', 9.0645],
        ]
    for fn, gt in lines:
        angle_pred = model.predict(fn)
        diff = abs(angle_pred-gt)
        if diff > 180:
            diff = 360 - diff
        print('gt:{:.4f}\t pred:{:.4f}\t diff:{:.4f}'.format(
            gt, angle_pred, diff))

def read_csv(csv_name):
    with open(csv_name) as f:
        csv_reader = csv.reader(f, delimiter=',')
        lines = list(csv_reader)
    for i in range(len(lines)):
        #print(lines[i])
        lines[i][1] = float(lines[i][1])
    return lines

def test_dataset(model):
    path_dst = '/home/Amoko/data/0813_noise_vector_5.4'
    
    csv_name = '/home/Amoko/data/0813_angle.csv'
    lines = read_csv(csv_name)
    for idx, (fn, gt) in enumerate(lines):
        angle_pred = model.predict(fn)
        diff = abs(angle_pred-gt)
        if diff > 180:
            diff = 360 - diff
        if diff < 10 or diff > 350:
            continue
        print(fn)
        print('{}\t gt:{:.4f}\t pred:{:.4f}\t diff:{:.4f}'.format(
            idx, gt, angle_pred, diff))
        
        shutil.copy(fn, path_dst)

if __name__ == '__main__':
    # 1 demo
    model_path = 'checkpoints/mobilenet_v2.3.4.pth'
    model_path = 'checkpoints/mobilenet_v2.0.2.pth'
    model_path = 'mobilenet_v2.4.1.pth'
    print(model_path)
    
    model = AngleInference(model_path)
    
    test_4img(model)
    #test_dataset(model)
    