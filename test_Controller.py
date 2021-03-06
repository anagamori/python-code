#!/usr/bin/env python2
# -*- coding: utf-8 -*-
"""
Created on Tue Feb 27 09:56:13 2018

@author: akiranagamori
"""

import numpy as np
import matplotlib.pyplot as plt
import os

default_path = '/Users/akira/Documents/Github/python-code/';  
save_path = '/Users/akira/Documents/Github/python-code/Data';  
        
os.chdir(save_path)
output = np.load('output.npy').item()
os.chdir(default_path)

Fs = 10000;
step = 1/float(Fs)

L0 = 3.0;
density = 1.06;   
mass = 0.0001;    
PCSA = (mass*1000)/(density*L0)
sigma = 22.4;
F0 = PCSA*sigma;

amp = 0.1
time_sim = np.arange(0,5,step)
Input = np.concatenate((np.zeros(1*Fs),amp/2*np.arange(0,2,step),amp*np.ones(5*Fs-3*Fs)),axis = 0)
F_target = F0*Input;      

Force = output['Tendon Force'];
delay_C = 50*Fs/1000;
K = 0.001;


Input_C = 0.0;
C_vec = np.zeros(len(time_sim));
dif_vec = np.zeros(len(time_sim));
force_vec = np.zeros(len(time_sim));

for t in xrange(len(time_sim)):
    if t > delay_C:
        difference = F_target[t]-force_vec[t-delay_C];
        Input_C += K*(F_target[t]-force_vec[t-delay_C]);
        C_vec[t] = Input_C;
        dif_vec[t] = difference
        
    force = Input_C*F0;
    force_vec[t] = force
    
plt.show()
plt.plot(F_target)    
plt.plot(force_vec)

