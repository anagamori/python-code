#!/usr/bin/env python2
# -*- coding: utf-8 -*-
"""
Created on Mon Mar 19 16:22:15 2018

@author: akira
"""

import numpy as np
from scipy import signal
from scipy import io
import matplotlib.pyplot as plt
import os

default_path = '/Users/akira/Documents/Github/python-code';  
save_path = '/Volumes/DATA2/SDN_Data';
Fs = 1000;

force_level = 100;

CoV = np.zeros(10);
Std = np.zeros(10);
mean_force = np.zeros(10);
mean_ISI = np.zeros(10);
CoV_ISI = np.zeros(10);
Std_ISI = np.zeros(10);
Pxx_all = np.zeros((10,2501)); # 100001 for 40000 Hz, 2501 for 1000 Hz
Force_all = np.zeros((10,5*Fs));
fileName_mean_force = "%s%s%s" % ('mean_force_noTendon_0_1_0_10_',str(force_level),'.npy')  
fileName_CoV = "%s%s%s" % ('CoV_noTendon_0_1_0_10_',str(force_level),'.npy')  
fileName_Std = "%s%s%s" % ('std_noTendon_0_1_0_10_',str(force_level),'.npy')  
fileName_mean_ISI = "%s%s%s" % ('mean_force_noTendon_0_1_0_10_',str(force_level),'.npy')  
fileName_Std_ISI = "%s%s%s" % ('std_ISI_noTendon_0_1_0_10_',str(force_level),'.npy')  
fileName_CoV_ISI = "%s%s%s" % ('CoV_ISI_noTendon_0_1_0_10_',str(force_level),'.npy')  
fileName_Force = "%s%s%s" % ('Force_noTendon_0_1_0_10_',str(force_level),'.mat')
trials = 0;
for i in range(force_level-10,force_level):    
#for i in range(30,31):    
    
    trialN = i;
    fileName = "%s%s%s" % ('output_SDN_noTendon_0_1_0_10_',str(trialN),'.npy')   
    os.chdir(save_path)
    output = np.load(fileName).item();
    Force = output['Muscle Force'][5*Fs:];
    mean_force[trials] = np.mean(Force);    
    CoV[trials] = np.std(Force)/np.mean(Force);    
    Std[trials] = np.std(Force);    
    os.chdir(default_path)
    f,Pxx = signal.periodogram(Force-np.mean(Force),Fs);
    Pxx_all[trials,:] = Pxx;
    Force_all[trials,:] = Force;
    
    spike_train = output['Spike Train'][0,5*Fs:];
    index = np.nonzero(spike_train);
    index_dif = np.diff(index);
    mean_ISI[trials] = np.mean(index_dif/float(Fs));
    Std_ISI[trials] = np.std(index_dif/float(Fs));
    CoV_ISI[trials] = Std_ISI[trials]/mean_ISI[trials]*100
    
    trials += 1;
    
    fig2 = plt.figure(2)
    ax2 = fig2.add_subplot(111);
    ax2.plot(output['Muscle Force']);

print (CoV)


os.chdir(save_path)
np.save(fileName_mean_force,mean_force)
np.save(fileName_CoV,CoV)
np.save(fileName_Std,Std)
np.save(fileName_mean_ISI,mean_ISI)
np.save(fileName_CoV_ISI,CoV_ISI)
np.save(fileName_Std_ISI,Std_ISI)
io.savemat(fileName_Force,mdict = {'Force':Force_all});
os.chdir(default_path)


fig1 = plt.figure(1)
ax1 = fig1.add_subplot(111);
ax1.plot(f,Pxx);
ax1.set_xlim([0,30]);



