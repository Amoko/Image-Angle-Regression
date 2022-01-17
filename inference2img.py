#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Sun Aug 15 15:58:59 2021

@author: Amoko
"""


import os
import time
import shutil
from test import AngleInference, get_zero_degree_img

def get_filenames(path_dir):
    filenames = os.listdir(path_dir)
    for i in range(len(filenames)):
        filenames[i] = os.path.join(path_dir, filenames[i])
    return filenames

if __name__ == '__main__':
    # 1 demo
    model_path = 'mobilenet_v2.2.3.pth'
    print(model_path)
    model = AngleInference(model_path)

    path_origin = '/home/Amoko/data/0902badcase'
    path_save = path_origin + '_adjusted'
    if not os.path.exists(path_save):
        os.mkdir(path_save)

    filenames = get_filenames(path_origin)
    print(len(filenames))
    for i in range(len(filenames)):
        fn = filenames[i]
        angle = model.predict(fn)
        print(angle)

        if i%1000==0:
            print(i, time.ctime())
            print(fn, angle)

        img = get_zero_degree_img(fn, angle)
        fn = 'adjusted.' + os.path.basename(fn)
        fn = os.path.join(path_save, fn)
        img.save(fn)
