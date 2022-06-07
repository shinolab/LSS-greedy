# -*- coding: utf-8 -*-
"""
Created on Mon Jun  6 15:27:50 2022

@author: Gin
"""

import os
import time
import math
import matplotlib.pyplot as plt
import timeit
import numpy as np
import mpl_toolkits.axes_grid1
from numba import jit, prange
from numba import c16, f8, i4, i8
import random


autd_num = 1
NUM_TRANS_X = 18 * autd_num
NUM_TRANS_Y = 14 * autd_num
#define the autd
TRANS_SIZE = 10.18
FREQUENCY = 40e3
Wave_len = 340 / FREQUENCY * 1000
# Wave_Number = 2 * np.pi  / Wave_len
Z_DIR = np.array([0, 0, 1])
z = 150.0
array_center = np.array([TRANS_SIZE * (NUM_TRANS_X - 1) / 2.0, TRANS_SIZE * (NUM_TRANS_Y - 1) / 2.0, 0])

#define transducers position
tran_pos_x = np.arange(0,NUM_TRANS_X*10.18,10.18)
tran_pos_y = np.arange(0,NUM_TRANS_Y*10.18,10.18)


#define observe surface
R = 100
X_RANGE = (array_center[0] - R / 2, array_center[0] + R / 2)
Y_RANGE = (array_center[1] - R / 2, array_center[1] + R / 2)
RESOLUTION = 1.0

os_x = np.arange(X_RANGE[0],X_RANGE[1],RESOLUTION)
os_y = np.arange(Y_RANGE[0],Y_RANGE[1],RESOLUTION)
os_z = z

@jit(c16[:,:,:](f8[:,:]))
def lss(c_list):
    n = len(c_list)

    phase = np.zeros((n, NUM_TRANS_X, NUM_TRANS_Y),dtype=np.complex128)
    for i in range(n):
        xx = c_list[i][0]
        yy = c_list[i][1]
        focal_pos = array_center + z * Z_DIR + np.array([-50 + xx, -50 + yy, 0])
        for x in range(NUM_TRANS_X):
            for y in range(NUM_TRANS_Y):
                now_d = np.sqrt((tran_pos_x[x] - focal_pos[0])**2+(tran_pos_y[y] - focal_pos[1])**2+focal_pos[2]**2)      
                phase[i][x][y] = np.exp(1j * (2 * np.pi * (now_d % Wave_len)/(Wave_len)))
    
    

    return phase

@jit(c16[:,:,:](c16[:,:,:]))
def gtop(phase):
    nfocus = np.shape(phase)[0]
    p_array = np.zeros((nfocus,100,100),dtype=np.complex128)
    
    for i in range(R):
        for j in range(R):
            for x in range(NUM_TRANS_X):
                for y in range(NUM_TRANS_Y):    
                    r = np.sqrt((os_x[i]-tran_pos_x[x])**2 + (os_y[j]-tran_pos_y[y])**2 + os_z**2)
                    p_each = 1 / (4 * np.pi) / r * np.exp(-1j * (2 * np.pi * (r % Wave_len)/(Wave_len)))
                    for num in range(nfocus):   
                        p_array[num][j][i] += p_each * phase[num][x][y]
               
    return p_array

def lssgreedy(c_list):
    n = len(c_list)
    
    p_array = np.zeros((n,100,100),dtype=np.complex128)
    phase = lss(c_list)
    p_array = gtop(phase)
    
    array_temp = np.zeros((n,100,100),dtype=np.complex128)    
    offset_list = np.zeros(n)
    offset_choice = np.arange(0,2*np.pi,np.pi/16)
    
    a = np.arange(1,n,1)
    random.shuffle(a)
    for i in a:
    # for i in range(1,n):
        p_max = 0
        p_tmp = 0
        offset_tmp = 0
        for j in range(len(offset_choice)):
            offset_list[i] = offset_choice[j]
            array = 0
            
            for k in range(n):
                if k == 0:
                    array_temp[k] = p_array[k]
                else:

                    array_temp[k] = p_array[k] * (np.exp(1j * offset_list[k]))
                array += array_temp[k]
            field_total = abs(array/n)

            focus_p = 0
            for f_num in range(n):
                
                focus_p += field_total[int(c_list[f_num][1])][int(c_list[f_num][0])]

            if focus_p > p_tmp:
                p_tmp = focus_p
                # print('p_tmp\t',p_tmp)
                offset_tmp = offset_choice[j]
                p_max = field_total
        offset_list[i] = offset_tmp
        
    for num in range(np.shape(phase)[0]):
        phase[num] = phase[num] * np.exp(1j * offset_list[num])
    return phase

def gspat(c_list):
    
    focal_pos = np.zeros((len(c_list),3))
    for focus_num in range(len(c_list)):
        focal_pos[focus_num] = array_center + z * Z_DIR + np.array([-50 + c_list[focus_num][0], -50 + c_list[focus_num][1], 0])

    #g.shape = M-focus  N-transducers
    g = np.zeros((len(c_list),NUM_TRANS_X*NUM_TRANS_Y),dtype=np.complex128)
    for num in range(len(c_list)):
        for x in range(NUM_TRANS_X):
            for y in range(NUM_TRANS_Y):
                r =np.sqrt((focal_pos[num][0] - tran_pos_x[x])**2+(focal_pos[num][1] - tran_pos_y[y])**2+(focal_pos[num][2])**2)
                g[num][x + y*NUM_TRANS_X] += 1 / r * np.exp(1j * 2 * np.pi * (r % Wave_len)/(Wave_len))
    # print('g:\t',g)
    
    g_con = np.conjugate(g)
    
    b = np.zeros((NUM_TRANS_X*NUM_TRANS_Y,len(c_list)),dtype=np.complex128)
    for num in range(len(c_list)):
        for x in range(NUM_TRANS_X):
            for y in range(NUM_TRANS_Y):
                numerator = g_con[num][x + y*NUM_TRANS_X]
                denominator = np.sum((np.abs(g[num]))**2)
                b[x + y*NUM_TRANS_X][num] = numerator / denominator
    
                
    iter = 100
    r = g.dot(b)
    tar_amp = np.ones((len(c_list),1),dtype=np.complex128) #focus p [len(focus)]
    for repeat in range(iter): 
        gamma = r.dot(tar_amp)
        tar_amp = gamma / np.abs(gamma)
    gamma = r.dot(tar_amp)
    tar_amp = gamma / np.abs(gamma)**2
    # tau = b.dot(tar_amp)/np.abs(b.dot(tar_amp))
    tau = b.dot(tar_amp) 
    phase = np.zeros((NUM_TRANS_X,NUM_TRANS_Y),dtype=np.complex128)
    for x in range(NUM_TRANS_X):
        for y in range(NUM_TRANS_Y):
            phase[x][y] += tau[x + y*NUM_TRANS_X]       
    phase = phase / np.max(np.abs(phase))

    return phase


@jit(c16[:,:](f8[:,:]))
def GS(c_list):

    R = 100
    
    p = np.zeros((R,R),dtype=np.complex128)
    # c_list = [[25,50],[75,50]]
    tar_amp = np.ones(len(c_list),dtype=np.complex128)
    

    focal_pos = np.zeros((len(c_list),3))
    for focus_num in range(len(c_list)):
        focal_pos[focus_num] = array_center + z * Z_DIR + np.array([-50 + c_list[focus_num][0], -50 + c_list[focus_num][1], 0])
    # print('focal_pos:\t',focal_pos)
    
    iter = 200
    
    for repeat in range(iter):
        g_vec = np.zeros((NUM_TRANS_X,NUM_TRANS_Y),dtype=np.complex128)
        
        for x in range(NUM_TRANS_X):
            for y in range(NUM_TRANS_Y):
                for num in range(len(c_list)):
                    r =np.sqrt((focal_pos[num][0] - tran_pos_x[x])**2+(focal_pos[num][1] - tran_pos_y[y])**2+(focal_pos[num][2])**2)
                    g_vec[x][y] += 1/r * np.exp(1j * 2 * np.pi * (r % Wave_len)/(Wave_len)) *  tar_amp[num]
        # print('g_vec:\t',g_vec)
        
        p_vec = g_vec/np.abs(g_vec)
        
        
        b_vec = np.zeros(len(c_list),dtype=np.complex128)
        for num in range(len(b_vec)):
            for x in range(NUM_TRANS_X):
                for y in range(NUM_TRANS_Y):
                    d = np.sqrt((focal_pos[num][0] - tran_pos_x[x])**2+(focal_pos[num][1] - tran_pos_y[y])**2+(focal_pos[num][2])**2)
                    b_vec[num] += p_vec[x][y] * (1/d * np.exp(-1j * 2 * np.pi * (d % Wave_len)/(Wave_len)))
        
        # print('b_vec:\t',b_vec)
        
        tar_amp = b_vec/np.abs(b_vec)
    
   
    # print('g_vec:\t',g_vec)

    p = np.zeros((R,R),dtype=np.complex128)
    for i in range(R):
        for j in range(R):
            for x in range(NUM_TRANS_X):
                for y in range(NUM_TRANS_Y):
                    # d = np.sqrt((tran_pos_x[x] - focal_pos[0])**2+(tran_pos_y[y] - focal_pos[1])**2+focal_pos[2]**2)
                    # r = pow((pow((os_x[i]-tran_pos_x[x]),2) + pow((os_y[j]-tran_pos_y[y]),2) + pow(os_z,2)),0.5)
                    r = np.sqrt((os_x[i]-tran_pos_x[x])**2 + (os_y[j]-tran_pos_y[y])**2 + os_z**2)
                    # p_each = np.exp(1j * Wave_Number * (d - r))
                    p_each = 1 / (4 * np.pi) / r * np.exp(1j * ( -2 * np.pi * (r % Wave_len)/(Wave_len))) * p_vec[x][y]
                    p[j][i] += p_each
    
    
    return p
            

@jit(c16[:,:](f8[:,:]))
def gtop2(phase):

    #calculate the pressure in observe surface
    
    p_array = np.zeros((100,100),dtype=np.complex128)
    
    for i in range(R):
        for j in range(R):
            for x in range(NUM_TRANS_X):
                for y in range(NUM_TRANS_Y):    
                    r = np.sqrt((os_x[i]-tran_pos_x[x])**2 + (os_y[j]-tran_pos_y[y])**2 + os_z**2)
                    p_each = (1/abs(r)) * np.exp(1j * (phase[x][y] -(2 * np.pi * (r % Wave_len)/(Wave_len))))
                    p_array[j][i] += p_each
               
    return p_array

def greedy(c_list):

    phase = np.zeros((NUM_TRANS_X, NUM_TRANS_Y),dtype=np.float64)
    phase_choice = np.arange(0,2*np.pi,np.pi/16)
    
    for x in range(NUM_TRANS_X):
        for y in range(NUM_TRANS_Y):
            p_tmp = 0
            phase_tmp = 0
            for phase_k in range(len(phase_choice)):
                phase[x][y] = phase_choice[phase_k]
                field = gtop2(phase)
                ave_p = 0
                for n in range(len(c_list)):
                    ave_p += field[int(c_list[n][1])][int(c_list[n][0])]
                if ave_p > p_tmp:
                    p_tmp = ave_p
                    phase_tmp = phase_choice[phase_k]
            phase[x][y] = phase_tmp

    p = gtop2(phase)
    return p