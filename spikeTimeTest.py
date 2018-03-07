#!/usr/bin/env python2
# -*- coding: utf-8 -*-
"""
Created on Tue Mar  6 15:53:09 2018

@author: akiranagamori
"""

import numpy as np
import matplotlib.pyplot as plt
import os

default_path = '/Users/akiranagamori/Documents/GitHub/python-code/';  
save_path = '/Users/akiranagamori/Documents/GitHub/python-code/Data';  
        
os.chdir(save_path)
output = np.load('output_temp.npy').item()
os.chdir(default_path)
MFR_MU = 8;
spike_time = output['spike_time']
FR = output['FR']
Fs = 1000;

index_vec = np.zeros(len(spike_time))
index1 = np.zeros(len(spike_time))
index2 = np.zeros(len(spike_time))
for i in xrange(len(spike_time)):
    if i > 1:
        index1[i] = FR[i] >= MFR_MU;
        index2[i] = FR[i-1] < MFR_MU;
        index = np.logical_and(FR[i] >= MFR_MU,FR[i-1] < MFR_MU);
        
        index_vec[i] = index;
        
    
plt.plot(index_vec)