#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Wed Aug  4 11:46:08 2021

@author: admin
"""

import os
import csv
import json
import numpy as np

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

def get_angle_from_json(fn):
    with open(fn) as f:
        data_dict = json.load(f)
        p_start, p_end = data_dict['shapes'][0]['points']
        p_start[1] *= -1
        p_end[1] *= -1
        
    v1 = [p_end[0] - p_start[0], p_end[1] - p_start[1]]
    angle = get_angle_from_vector(v1)
    return angle

def get_vector_from_json(fn):
    with open(fn) as f:
        data_dict = json.load(f)
        p_start, p_end = data_dict['shapes'][0]['points']
        p_start[1] *= -1
        p_end[1] *= -1
        
    v = [p_end[0] - p_start[0], p_end[1] - p_start[1]]
    v = np.array(v)
    norm = np.linalg.norm(v)
    v /= norm
    return v

def write_final_csv(path_dir):
    csv_name = path_dir + '.csv'
    filenames = os.listdir(path_dir)
    lines = []
    for fn in filenames:
        if fn[-4:] == 'json':
            fn = os.path.join(path_dir, fn)
            angle = get_angle_from_json(fn)
            fn = fn[:-4] + 'png'
            line = [fn, angle]
            if os.path.exists(fn):
                lines.append(line)
            else:
                print('img not exists', fn)
    
    with open(csv_name, mode='w') as f:
        writer = csv.writer(f)
        for line in lines:
            writer.writerow(line)
            
def write_final_csv_vector(path_dir):
    csv_name = path_dir + '_vector.csv'
    filenames = os.listdir(path_dir)
    lines = []
    for fn in filenames:
        if fn[-4:] == 'json':
            fn = os.path.join(path_dir, fn)
            angle = get_angle_from_json(fn)
            vector = get_vector_from_json(fn)
            fn = fn[:-4] + 'png'
            line = [fn, angle, vector[0], vector[1]]
            if os.path.exists(fn):
                lines.append(line)
            else:
                print('img not exists', fn)
    
    with open(csv_name, mode='w') as f:
        writer = csv.writer(f)
        for line in lines:
            writer.writerow(line)

if __name__ == '__main__':

    fn = '0805_angle/0_1_I100710709_11_0.json'
    fn = 'raw/wrong286_labeled/0_1_I105173993_696_2_0.json'
    fn = 'raw/wrong286_labeled/0_1_I105344732_1_2_0.json'
    fn = 'raw/wrong286_labeled/0_1_I93237264_1_6_0.json'
    fn = 'raw/wrong286_labeled/0_1_I94079959_2_0.json'
    angle = get_angle_from_json(fn)
    vector = get_vector_from_json(fn)
    print(fn)
    print(angle, vector)
    
    path_dir = '/home/Amoko/data/val_1_good_filter_bi_1of2'
    path_dir = '/home/Amoko/data/0804_angle'
    path_dir = '/home/Amoko/data/0805_angle'
    path_dir = '/home/Amoko/data/0813_angle'
    #path_dir = '/home/Amoko/data/0813_angle_fine'
    
    #write_final_csv(path_dir)
    #write_final_csv_vector(path_dir)
    

    
    
    


