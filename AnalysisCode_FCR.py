#!/usr/bin/env python2
# -*- coding: utf-8 -*-
"""
Created on Fri Mar  9 17:55:33 2018

@author: akira
"""
import numpy as np
from scipy import signal
#from scipy import io
import matplotlib.pyplot as plt
import os

default_path = '/Users/akira/Documents/Github/python-code';  
save_path = '/Volumes/DATA2/FCR_Data';
Fs = 10000;

trials = range(0,10);
CoV = np.zeros(len(trials))
CoV_2 = np.zeros(len(trials))
Pxx_All = np.zeros((len(trials),50001))
Pxx_All_2 = np.zeros((len(trials),50001))

for i in xrange(len(trials)):  
    trialN = trials[i]+450;
    trialN_2 = trials[i]+460;
    fileName = "%s%s%s" % ('output_FCR_',str(trialN),'.npy')  
    fileName_2 = "%s%s%s" % ('output_FCR_',str(trialN_2),'.npy')       
    os.chdir(save_path)
    output = np.load(fileName).item()
    Force = output['Tendon Force'][5*Fs:];
    CoV[i] = np.std(Force)/np.mean(Force);
    output = np.load(fileName_2).item()
    Force_2 = output['Tendon Force'][5*Fs:];
    CoV_2[i] = np.std(Force_2)/np.mean(Force_2);
    os.chdir(default_path)
    
    f,Pxx = signal.periodogram(Force-np.mean(Force),Fs);
    f_2,Pxx_2 = signal.periodogram(Force_2-np.mean(Force_2),Fs);
    Pxx_All[i,:] = Pxx;
    Pxx_All_2[i,:] = Pxx_2;
    
fig1 = plt.figure(1)
ax1 = fig1.add_subplot(111);
ax1.plot(f,np.mean(Pxx_All,0));
ax1.plot(f,np.mean(Pxx_All_2,0));
ax1.set_xlim([0,30]);

fig2 = plt.figure(2)
ax2 = fig2.add_subplot(111);
ax2.boxplot([CoV,CoV_2])



#freqs = np.arange(0,20,0.1)
#windowSize = 2.5*Fs;
#overlap = 0;
#(freqs,coherence) = mscohere(signal1,signal2,Fs,freqs,windowSize,overlap);

#plt.figure(1)
#plt.plot(freqs,coherence)