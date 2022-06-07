# -*- coding: utf-8 -*-
"""
Created on Mon Jun  6 15:44:30 2022

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




def plotphase(NUM_TRANS_X,NUM_TRANS_Y,TRANS_SIZE,phase):
    trans_x = []
    trans_y = []
    trans_phase = []
    print(phase)
    for i in range(NUM_TRANS_Y):
        for j in range(NUM_TRANS_X):
            trans_x.append(j*10.18)
            trans_y.append(i*10.18)
            trans_phase.append(phase[j][i])
    
    print(phase)
    dpi = 72
    fig = plt.figure(figsize=(6, 6), dpi=dpi)
    axes = fig.add_subplot(111, aspect='equal')
    scat = axes.scatter(trans_x, trans_y, c=trans_phase, cmap='jet', s=0,
                        marker='o', vmin=0, vmax=2*math.pi,
                        clip_on=False, linewidths=0)
    fig.canvas.draw()
    r_pix = axes.transData.transform((TRANS_SIZE / 2, TRANS_SIZE / 2)) - axes.transData.transform((0, 0))
    sizes = (2.0*r_pix*72/fig.dpi)**2
    scat.set_sizes(sizes)
    divider = mpl_toolkits.axes_grid1.make_axes_locatable(axes)
    cax = divider.append_axes(position='right', size='5%', pad='3%')
    cb = fig.colorbar(scat, cax=cax)
    cb.set_ticks(np.linspace(0,2*np.pi,3))
    cb.set_ticklabels( ('0', 'π', '2π'))
    axes.set_xticks([])
    axes.set_yticks([])
    plt.xticks([])
    plt.yticks([])
    plt.axis('off')
    plt.show()          

def plotfield(p):
   
    fig = plt.figure(figsize=(6, 6), dpi=300)   
    axes = fig.add_subplot(111, aspect='equal')
    # heat_map = axes.pcolor(abs(p), cmap='jet', vmin=0, vmax=0.5)
    heat_map = axes.pcolor(abs(p), cmap='jet')
    ticks_step=10.0
    resolution = RESOLUTION
    x_label_num = int(math.floor((X_RANGE[1] - X_RANGE[0]) / ticks_step)) + 1
    y_label_num = int(math.floor((Y_RANGE[1] - Y_RANGE[0]) / ticks_step)) + 1
    x_labels = [X_RANGE[0] + ticks_step * i for i in range(x_label_num)]
    y_labels = [Y_RANGE[0] + ticks_step * i for i in range(y_label_num)]
    x_ticks = [ticks_step / resolution * i for i in range(x_label_num)]
    y_ticks = [ticks_step / resolution * i for i in range(y_label_num)]
    axes.set_xticks(np.array(x_ticks) + 0.5, minor=False)
    axes.set_yticks(np.array(y_ticks) + 0.5, minor=False)
    axes.set_xticklabels(x_labels, minor=False, fontsize=18)
    axes.set_yticklabels(y_labels, minor=False, fontsize=18)
    ax = plt.gca() # grab the current axis 
    plt.xticks([])
    plt.yticks([])
    # plt.axis('off')
    ax.set_xticks([0,10,20,30,40,50,60,70,80,90,100]) # choose which x locations to have ticks
    ax.set_xticklabels([-50,-40,-30,-20,-10,0,10,20,30,40,50]) # set the labels to display at those ticks
    ax.set_yticks([0,10,20,30,40,50,60,70,80,90,100]) # choose which x locations to have ticks
    ax.set_yticklabels([-50,-40,-30,-20,-10,0,10,20,30,40,50]) # set the labels to display at those ticks        
    # ax.set_ylabel(u'[mm]', loc='top')
    # ax.set_xlabel(u'[mm]', loc='right') 
    ax.set_ylabel(u'y[mm]', fontsize=14)
    ax.set_xlabel(u'x[mm]', fontsize=14) 
    divider = mpl_toolkits.axes_grid1.make_axes_locatable(axes)
    cax = divider.append_axes('right', '5%', pad='3%')
    
    cbar = fig.colorbar(heat_map, cax=cax)
    cbar.ax.tick_params(labelsize='16')
    cbar.set_label('Sound pressure', size=18)
    # bounds = ['0.00','0.100','0.200','0.300','0.400','0.500']
    # cbar.set_ticks([0,0.1,0.2,0.3,0.4,0.5])
    # cbar.set_ticklabels(bounds)

    plt.tight_layout()
    # num = 1
    # plt.savefig('C:/Codes/AUTDNN/fieldplot/field {0}.jpg'.format(num),bbox_inches='tight',dpi=fig.dpi,pad_inches=0.0)
    # plt.savefig(os.path.join(field_root,field_name + ' {0}.jpg'.format(num)),bbox_inches='tight',dpi=fig.dpi,pad_inches=0.0)
    # plt.savefig(os.path.join('gs.pdf'),bbox_inches='tight',dpi=fig.dpi,pad_inches=0.0)
    # plt.close(fig)
    plt.show()
    