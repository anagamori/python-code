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

default_path = '/Users/akiranagamori/Documents/GitHub/python-code/';  
save_path = '/Users/akiranagamori/Documents/GitHub/python-code/Data';  
        
os.chdir(save_path)
output = np.load('output_FCR.npy').item()
os.chdir(default_path)

Fs = 10000;
meanForce = np.mean(output['Tendon Force'][4*Fs:])
CoVForce = np.std(output['Tendon Force'][4*Fs:])/meanForce

Force = output['Tendon Force'][4*Fs:];
f,Pxx = signal.periodogram(Force-np.mean(Force),Fs);

fig1 = plt.figure()
plt.plot(output['ND'])

fig2 = plt.figure()
ax2 = fig2.add_subplot(111);
ax2.plot(f,Pxx);
ax2.set_xlim([0,30]);
