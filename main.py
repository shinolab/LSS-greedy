# -*- coding: utf-8 -*-
"""
Created on Mon Jun  6 15:18:23 2022

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

import calfield
import methods
import plot

focus_num = 12
autd_num = 1
NUM_TRANS_X = 18 * autd_num
NUM_TRANS_Y = 14 * autd_num

def dc(x0,y0,r,focus_num):

    focus_list_temp = np.zeros((focus_num,3))
    
    for focus_list_i in range(focus_num):

        focus_list_temp[focus_list_i][0]=x0 + round(r * np.cos((2 * np.pi / focus_num) * (focus_list_i-1)))
        focus_list_temp[focus_list_i][1]=y0 + round(r * np.sin((2 * np.pi / focus_num) * (focus_list_i-1)))

    # dividecircle=focus_list_temp
    return focus_list_temp


def star():
    pos1_temp = dc(50,50,43,5)
    pos1 = np.zeros(np.shape(pos1_temp))
    for i in range(np.shape(pos1_temp)[0]):
        pos1[i][0] = pos1_temp[i][1]
        pos1[i][1] = pos1_temp[i][0]
    pos2_temp = dc(50,50,16,5)
    pos2 = np.zeros(np.shape(pos2_temp))
    for i in range(np.shape(pos2_temp)[0]):
        pos2[i][0] = pos2_temp[i][1]
        pos2[i][1] = pos2_temp[i][0]
    for i in range(np.shape(pos2)[0]):
        
        pos2[i][1] = 100 - pos2[i][1]
        
    pos3 = np.array([[50,50,0]])
    focus_list_temp = np.concatenate((pos1,pos2,pos3),axis=0)
    
    return focus_list_temp

c_list = dc(50,50,30,focus_num)
# c_list = star()



# plss = calfield.callss(methods.lss(c_list))
plssgreedy = calfield.callss(methods.lssgreedy(c_list))
# pgspat = calfield.calgs(methods.gspat(c_list))

# plot.plotfield(plss)
plot.plotfield(plssgreedy)
# plot.plotfield(pgspat)

