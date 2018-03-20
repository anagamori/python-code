#!/usr/bin/env python2
# -*- coding: utf-8 -*-
"""
Created on Wed Mar  7 15:08:47 2018

@author: akiranagamori
"""

import numpy as np
import matplotlib.pyplot as plt
import os
from scipy import signal

default_path = '/Users/akira/Documents/Github/python-code';  
save_path = '/Volumes/DATA2/Synergists_Data';  

#trials = [0,1,2,4,5,6,7,8,9];
trials = range(0,100);
CoVForce = np.zeros(len(trials));
ForceRatio = np.zeros(len(trials));
for i in xrange(len(trials)):  
    trialN = trials[i];
    fileName = "%s%s%s" % ('output_Synergists_',str(trialN),'.npy')       
    os.chdir(save_path)
    output = np.load(fileName).item()
    os.chdir(default_path)
    
    Fs = 10000;
    meanForce = np.mean(output['Total Force'][5*Fs:])
    CoVForce[i] = np.std(output['Total Force'][5*Fs:])/meanForce
    
    meanForce1 = np.mean(output['Tendon Force'][5*Fs:])
    meanForce2 = np.mean(output['Tendon Force 2'][5*Fs:])
    if meanForce1 > meanForce2:
        ForceRatio[i] = meanForce2/meanForce1;
    else:
        ForceRatio[i] = meanForce1/meanForce2;
    Force = output['Total Force'][4*Fs:];
    
    f,Pxx = signal.periodogram(Force-np.mean(Force),Fs);
    
#    signal1 = output['Spike Train'][0,5*Fs:]
#    signal2 = output['Spike Train'][11,5*Fs:]
#    f_C,Cxy = signal.coherence(signal1-np.mean(signal1),signal2-np.mean(signal2),Fs);
    
    fig1 = plt.figure(1)
    plt.plot(output['Total Force'])   
    
    fig2 = plt.figure(2)
    ax2 = fig2.add_subplot(111);
    ax2.plot(f,Pxx);
    ax2.set_xlim([0,30]);
    
#    fig3 = plt.figure(3)
#    ax3 = fig3.add_subplot(111);
#    ax3.plot(f_C,Cxy);
#    ax3.set_xlim([0,100]);
    
plt.show()

plt.figure(4)
plt.hist(CoVForce[CoVForce<0.1]*100)

plt.figure(5)
plt.hist(ForceRatio)

