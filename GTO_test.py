#!/usr/bin/env python2
# -*- coding: utf-8 -*-
"""
Created on Sun Feb 25 12:26:21 2018

@author: akiranagamori
"""
import numpy as np
import matplotlib.pyplot as plt
import os

def GTOOutput(FR_Ib,FR_Ib_temp,x_GTO,Force,index):
    G1 = 60;
    G2 = 4;
    num1 = 1.7;
    num2 = -3.3997;
    num3 = 1.6997;
    den1 = 1.0;
    den2 = -1.9998;
    den3 = 0.9998;
            
    x_GTO[index] = G1*np.log(Force/float(G2)+1);
          
    FR_Ib_temp[index] = (num3*x_GTO[index-2] + num2*x_GTO[index-1] + num1*x_GTO[index] - 
              den3*FR_Ib_temp[index-2] - den2*FR_Ib_temp[index-1])/den1;
    FR_Ib[index] = FR_Ib_temp[index];
    if FR_Ib[index]<0:
        FR_Ib[index] = 0;
    
    return (FR_Ib,FR_Ib_temp,x_GTO)

Fs = 10000;
step = 1/float(Fs)
time_sim = np.arange(0,5,step)

G1 = float(60);
G2 = float(4);
num1 = float(1.7);
num2 = float(-3.399742022978487);
num3 = float(1.699742026978047);
den1 = float(1.0);
den2 = float(-1.999780020198665);
den3 = float(0.999780024198225);
    
x_GTO = np.zeros(len(time_sim),dtype=np.float128);
FR_Ib_temp = np.zeros(len(time_sim),dtype=np.float128);
FR_Ib = np.zeros(len(time_sim),dtype=np.float128);
    
default_path = '/Users/akiranagamori/Documents/GitHub/python-code/';  
save_path = '/Users/akiranagamori/Documents/GitHub/python-code/Data';  
        
os.chdir(save_path)
output = np.load('output.npy').item()
os.chdir(default_path)

# Force = output['Tendon Force'];
amp = 1.0;
Force = np.concatenate((np.zeros(1*Fs),amp/2*np.arange(0,2,step),amp*np.ones(5*Fs-3*Fs)),axis = 0)

for t in xrange(len(time_sim)):
    if t > 3:
        x_GTO[t] = G1*np.log(Force[t]/float(G2)+1);
        FR_Ib_temp[t] = (num3*x_GTO[t-2] + num2*x_GTO[t-1] + num1*x_GTO[t] - 
                  den3*FR_Ib_temp[t-2] - den2*FR_Ib_temp[t-1])/float(den1);
        FR_Ib[t] = FR_Ib_temp[t];
        if FR_Ib[t]<0:
            FR_Ib[t] = 0;
        
    
plt.plot(FR_Ib)



    

