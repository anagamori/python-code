#!/usr/bin/env python2
# -*- coding: utf-8 -*-
"""
Created on Fri Mar  9 09:18:44 2018

@author: akira
"""
import numpy as np
from scipy import signal
from scipy import io
import matplotlib.pyplot as plt
import os

def mscohere(signal1,signal2,Fs,freqs,windowSize,overlap):
    taper = signal.gaussian(windowSize,(windowSize-1)/(2*2.5));
    taper = taper/np.sum(taper);
    windowTime = np.arange(windowSize);
    windowTime = windowTime/float(Fs);
    startPoints = np.arange(0,len(signal1),windowSize-overlap);
    index_delete = startPoints[startPoints>len(signal1)-windowSize+1];
    if index_delete.size != 0:
        np.delete(startPoints,index_delete,None);
    
    
    dataSegment1 = np.zeros((len(startPoints),int(windowSize)));
    dataSegment2 = np.zeros((len(startPoints),int(windowSize)));
    mscoh_temp = np.zeros((len(startPoints),len(freqs)));
    
    for i in range(len(startPoints)):
        dataSegment1[i,:] = signal1[int(startPoints[i]):int(startPoints[i])+int(windowSize)];
        dataSegment2[i,:] = signal2[int(startPoints[i]):int(startPoints[i])+int(windowSize)];
        for f in range(len(freqs)):
            complex_sine = np.exp(2j*np.pi*freqs[f]*windowTime);
            kernel = np.multiply(complex_sine,taper);
            complexTFR1 = 2*np.convolve(dataSegment1[i,:],kernel);
            complexTFR2 = 2*np.convolve(dataSegment2[i,:],kernel);
            complexTFR1 = complexTFR1[:len(kernel)];
            complexTFR2 = complexTFR2[:len(kernel)];
            phaseLocking = np.exp(1j*(np.angle(complexTFR1)-np.angle(complexTFR2)));
            amp1 = np.abs(complexTFR1);
            amp2 = np.abs(complexTFR2);
            num = np.power(np.abs(np.sum(np.multiply(amp1,amp2,phaseLocking))),2);
            den = np.sum(np.power(amp1,2))*np.sum(np.power(amp2,2));
            mscoh_temp[i,f] = num/den;
            
    coherence = np.mean(mscoh_temp,0);
    return (freqs,coherence)

default_path = '/Users/akira/Documents/Github/python-code';  
save_path = '/Volumes/DATA2/Synergists_Data';  
trialN = 5;
fileName = "%s%s%s" % ('output_Synergists_',str(trialN),'.npy')       
os.chdir(save_path)
output = np.load(fileName).item()
os.chdir(default_path)

plt.plot(output['Total Force'])

Fs = 10000;

nMax = 150;
for i in range(0,50):
    index1 = np.random.randint(0,nMax,1);
    index2 = np.random.randint(0,nMax,1);
    signal1_temp = output['Spike Train'][index1,5*Fs:]
    signal1_temp = signal1_temp - np.mean(signal1_temp);
    signal2_temp = output['Spike Train 2'][index2,5*Fs:]   
    signal2_temp = signal2_temp - np.mean(signal2_temp);
    if i == 0:
        signal1 = signal1_temp;
        signal2 = signal2_temp;
    else:
#        signal1 = np.append(signal1,signal1_temp);
#        signal2 = np.append(signal2,signal2_temp);
        signal1 = signal1 + signal1_temp;
        signal2 = signal2 + signal2_temp;
        

#signal1 = output['ND'][5*Fs:];
#signal2 = output['ND_2'][5*Fs:] 
signal_mat = {'signal1': signal1, 'signal2': signal2}
io.savemat('signal.mat',signal_mat)

#freqs = np.arange(0,20,0.1)
#windowSize = 2.5*Fs;
#overlap = 0;
#(freqs,coherence) = mscohere(signal1,signal2,Fs,freqs,windowSize,overlap);

#plt.figure(1)
#plt.plot(freqs,coherence)