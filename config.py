#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Tue Aug  3 20:29:37 2021

@author: Amoko
"""

import os

class MyConfig():
    def __init__(self):
        # 1 path
        self.csv_path = '/home/Amoko/data/0804_angle.csv'
        self.csv_path = '/home/Amoko/data/0813_angle_vector.csv'
        self.csv_path = '/home/Amoko/data/angle_vector/0901_angle_vector.csv'
        self.csv_path = '/home/Amoko/data/angle_vector/0902_angle_vector.csv'
        self.csv_path = '/home/Amoko/data/angle_vector/csv/0902_angle_vector_cleaned6891.csv'

        self.old_checkpoint = 'checkpoints/mobilenet_v2.3.3.pth'
        self.old_checkpoint = None


        # 2 train
        self.LR = 1e-4
        self.BATCH_SIZE = 128
        self.IMG_SIZE = (224, 224)
        
        self.amp = False
        
        self.degree_scale = 180
        
        self.sample_type = 'continuous'
        self.sample_type = 'discrete'

        self.sample_continuous = [-90, 90]
        self.sample_continuous = [-179, 179]
        
        self.sample_discrete = [-90,-45,-25,-5,30,60,95]
        self.sample_discrete = list(range(-175, 175, 2))

        
        
