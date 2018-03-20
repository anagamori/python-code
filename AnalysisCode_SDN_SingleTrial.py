#!/usr/bin/env python2
# -*- coding: utf-8 -*-
"""
Created on Mon Mar 12 14:16:41 2018

@author: akira
"""

import numpy as np
from scipy import signal
from scipy import io
import matplotlib.pyplot as plt
import os

default_path = '/Users/akira/Documents/Github/python-code';  
save_path = '/Volumes/DATA2/SDN_Data';
Fs = 10000;

force_level = 10;

CoV = np.zeros(10);
Std = np.zeros(10);
mean_force = np.zeros(10);
mean_ISI = np.zeros(10);
CoV_ISI = np.zeros(10);
Std_ISI = np.zeros(10);
Pxx_all = np.zeros((10,200001)); # 100001 for 40000 Hz, 2501 for 1000 Hz
Force_all = np.zeros((10,5*Fs));
fileName_mean_force = "%s%s%s" % ('mean_force_0_1_0_0_',str(force_level),'.npy')  
fileName_CoV = "%s%s%s" % ('CoV_0_1_0_0_',str(force_level),'.npy')  
fileName_Std = "%s%s%s" % ('std_0_1_0_',str(force_level),'.npy')  
fileName_mean_ISI = "%s%s%s" % ('mean_force_0_1_0_0_',str(force_level),'.npy')  
fileName_Std_ISI = "%s%s%s" % ('std_ISI_0_1_0_0_',str(force_level),'.npy')  
fileName_CoV_ISI = "%s%s%s" % ('CoV_ISI_0_1_0_0_',str(force_level),'.npy')  
fileName_Force = "%s%s%s" % ('Force_0_1_0_0_',str(force_level),'.mat')
trials = 0;
 
    
trialN = 0;
fileName = "%s%s%s" % ('output_SDN_SingleTrial_0_1_0_0_',str(trialN),'.npy')   
os.chdir(save_path)
output = np.load(fileName).item();
Force = output['Tendon Force'][5*Fs:];
mean_force = np.mean(Force);    
CoV = np.std(Force)/np.mean(Force);    
Std = np.std(Force);    
os.chdir(default_path)
f,Pxx = signal.periodogram(Force-np.mean(Force),Fs);

spike_train = output['Spike Train'][0,5*Fs:];
index = np.nonzero(spike_train);
index_dif = np.diff(index);
mean_ISI = np.mean(index_dif/float(Fs));
Std_ISI = np.std(index_dif/float(Fs));
CoV_ISI = Std_ISI/mean_ISI*100

trials += 1;

fig2 = plt.figure(2)
ax2 = fig2.add_subplot(111);
ax2.plot(output['Tendon Force']);

print (CoV)


#os.chdir(save_path)
#np.save(fileName_mean_force,mean_force)
#np.save(fileName_CoV,CoV)
#np.save(fileName_Std,Std)
#np.save(fileName_mean_ISI,mean_ISI)
#np.save(fileName_CoV_ISI,CoV_ISI)
#np.save(fileName_Std_ISI,Std_ISI)
#io.savemat(fileName_Force,mdict = {'Force':Force_all});
#os.chdir(default_path)


fig1 = plt.figure(1)
ax1 = fig1.add_subplot(111);
ax1.plot(f,Pxx);
ax1.set_xlim([0,30]);



