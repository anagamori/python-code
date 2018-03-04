#!/usr/bin/env python2
# -*- coding: utf-8 -*-
"""
Created on Sun Mar  4 08:37:34 2018

@author: akiranagamori
"""

import numpy as np
import matplotlib.pyplot as plt
import os

def twitch_function(Af,Lce,CT,RT,Fs):
    T1 = CT*np.power(Lce,2) + (CT/2)*Af;
    T2_temp = (RT + (RT/2)*Af)/Lce;
    T2 = T2_temp/1.68;
    t_twitch = np.arange(0,2,1/float(Fs));
    f_1 = np.multiply(t_twitch/T1,np.exp(1-t_twitch/T1));
    f_2 = np.multiply(t_twitch/T2,np.exp(1-t_twitch/T2));
    twitch = np.append(f_1[0:int(np.round(T1*Fs+1))],f_2[int(np.round(T2*Fs+1)):]);
    twitch = twitch[0:1*Fs];
    return twitch
    

default_path = '/Users/akiranagamori/Documents/GitHub/python-code/';  
save_path = '/Users/akiranagamori/Documents/GitHub/python-code/Data';  
        
os.chdir(save_path)
output = np.load('output_temp.npy').item()
os.chdir(default_path)

Fs = 1000;

#Af = output['Af'][-1];
Lce = output['Muscle Length'][-1];
CT = 0.0632;
RT = 0.0632;

spikeTrain = output['Spike Train'][0,:];
Pi = 3.4516e-04;
force = np.zeros(len(spikeTrain));

for t in xrange(len(spikeTrain)):
    if spikeTrain[t] == 1:
        Af = output['Af'][t];
        FF = output['FF'][t];
        twitch_temp = twitch_function(Af,Lce,CT,RT,Fs);
        twitch = Pi*twitch_temp*FF;
        if len(spikeTrain)-t >= len(twitch): 
            force[t:t+len(twitch)] = force[t:t+len(twitch)] + twitch;
        else:               
            force[t:] = force[t:] + twitch[:len(spikeTrain)-t];
        
plt.plot(force)    
    

