#!/usr/bin/env python2
# -*- coding: utf-8 -*-
"""
Created on Mon Mar 12 09:35:35 2018

@author: akira
"""

import numpy as np
from scipy import signal
from scipy import io
import matplotlib.pyplot as plt
import os

default_path = '/Users/akira/Documents/Github/python-code';  
save_path = '/Volumes/DATA2/FCR_Data';
Fs = 10000;

trials = 0;

trialN = trials;
fileName = "%s%s%s" % ('output_FCR_Test_',str(trialN),'.npy')   
os.chdir(save_path)
output = np.load(fileName).item()
Force = output['Tendon Force'][7*Fs:];
CoV = np.std(Force)/np.mean(Force);
os.chdir(default_path)

print (CoV)
f,Pxx = signal.periodogram(Force-np.mean(Force),Fs);

nMax = 150;
for i in range(0,50):
    index1 = np.random.randint(0,nMax,1);
    index2 = np.random.randint(0,nMax,1);
    signal1_temp = output['Spike Train'][index1,5*Fs:]
    signal1_temp = signal1_temp - np.mean(signal1_temp);
    signal2_temp = output['Spike Train'][index2,5*Fs:]   
    signal2_temp = signal2_temp - np.mean(signal2_temp);
    if i == 0:
        signal1 = signal1_temp;
        signal2 = signal2_temp;
    else:
        signal1 = np.append(signal1,signal1_temp);
        signal2 = np.append(signal2,signal2_temp);
        
signal_mat = {'signal1': signal1, 'signal2': signal2}
io.savemat('signal.mat',signal_mat)

fig1 = plt.figure(1)
ax1 = fig1.add_subplot(111);
ax1.plot(f,Pxx);
ax1.set_xlim([0,30]);

fig2 = plt.figure(2)
ax2 = fig2.add_subplot(111);
ax2.plot(output['Tendon Force']);
