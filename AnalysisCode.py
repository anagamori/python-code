#!/usr/bin/env python2
# -*- coding: utf-8 -*-
"""
Created on Fri Feb 23 10:24:55 2018

@author: akiranagamori
"""

import numpy as np
import matplotlib.pyplot as plt
import os
from scipy import signal

default_path = '/Users/akira/Documents/Github/python-code/';  
save_path = '/Users/akira/Documents/Github/python-code/Data';  
        
os.chdir(save_path)
output = np.load('output1.npy').item()
os.chdir(default_path)

Fs = 40000;
step = 1/float(Fs);
Force = output['Tendon Force'][4*Fs:];
ND = output['ND'][4*Fs:];
meanND = np.mean(ND);
meanForce = np.mean(Force)
CoVForce = np.std(Force)/meanForce
f,Pxx = signal.periodogram(Force-np.mean(Force),Fs);

# interspike-intervals
spikeTrain = output['Spike Train'][3,4*Fs:];
index = np.nonzero(spikeTrain);
index_dif = np.diff(index);
mean_ISI = np.mean(index_dif)/1000*10;
sd_ISI = np.std(index_dif);
CoV = sd_ISI/np.mean(index_dif)*100
print(CoV)

fig1 = plt.figure()
plt.plot(output['Muscle Velocity'])

#fig2 = plt.figure()
#ax2 = fig2.add_subplot(111);
#ax2.plot(f,Pxx);
#ax2.set_xlim([0,30]);

#fig3 = plt.figure()


