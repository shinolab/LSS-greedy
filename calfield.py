# -*- coding: utf-8 -*-
"""
Created on Mon Jun  6 15:22:12 2022

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


@jit(c16[:,:](c16[:,:,:]))
def callss(phase):
    
    phase_new = np.zeros((NUM_TRANS_X, NUM_TRANS_Y),dtype=np.complex128)

    phase_new = phase.sum(axis = 0)

    phase_new = phase_new / np.max(np.abs(phase_new))
    #calculate the pressure in observe surface
    p = np.zeros((R,R),dtype=np.complex128)
    
    for i in range(R):
        for j in range(R):
            for x in range(NUM_TRANS_X):
                for y in range(NUM_TRANS_Y):
                    
                    r = np.sqrt((os_x[i]-tran_pos_x[x])**2 + (os_y[j]-tran_pos_y[y])**2 + os_z**2)
                    # p_each = np.exp(1j * Wave_Number * (d - r))
                    p_each = 1 / (4 * np.pi) / r * np.exp(-1j * (2 * np.pi * (r % Wave_len)/(Wave_len))) * phase_new[x][y]
                    p[j][i] += p_each

    return p



@jit(c16[:,:](c16[:,:]))
def calgs(phase):
    #calculate the pressure in observe surface
    p = np.zeros((R,R),dtype=np.complex128)
    for i in range(R):
        for j in range(R):
            for x in range(NUM_TRANS_X):
                for y in range(NUM_TRANS_Y):                 
                    r = np.sqrt((os_x[i]-tran_pos_x[x])**2 + (os_y[j]-tran_pos_y[y])**2 + os_z**2)
                    p_each = 1 / (4 * np.pi) / r * np.exp(1j * (2 * np.pi * (r % Wave_len)/(Wave_len))) * phase[x][y]
                    p[j][i] += p_each

    return p