#!/usr/bin/env python2
# -*- coding: utf-8 -*-
"""
Created on Wed Jan 17 14:42:25 2018

@author: akiranagamori
"""

import numpy as np
import matplotlib.pyplot as plt

Fs = 1000
step = 1/float(Fs)
amp = 0.2
time = np.arange(0,5,step)
U = np.concatenate((np.zeros(1*Fs),amp/float(2)*np.arange(0,2,step),amp*np.ones(5*Fs-3*Fs)),axis = 0)
# plt.plot(t,U)
# plt.show()
N = 120
i = np.arange(1,N+1,1);

RR = 30;
a = np.log(RR)/float(N)
RTE = np.exp(a*i)
MFR = 8;
g_e = 1;
PFR1 = 35;
PFRD = 10;
RTEn = np.exp(a*N)
PFR = PFR1 - PFRD*np.divide(RTE,RTEn)
PFRn = PFR1 - PFRD
Emax = RTEn + (PFRn - MFR)/float(g_e)
cv = 0.1;
RP = 100;
b = np.log(RP)/N
P = np.exp(b*i)
T_L = 90;
RT = 3;
c = np.log(RP)/np.log(RT)
temp1 = np.divide(1,P)
T = T_L*np.power(temp1,1/float(c))/1000

t_twitch = np.arange(0,1,step)
twitch = np.zeros((N,len(t_twitch)))

for j in range(1,N):
    twitch[j,:] = P[j]/float(T[j])*np.multiply(t_twitch,np.exp(1-t_twitch/T[j]))

output_FR = np.zeros((N,len(time)))
spike_time = np.zeros(N)
spike_train = np.zeros((N,len(time)))
force = np.zeros((N,len(time)))

E = Emax*U

# %%
for t in range(1,len(time)):
    FR = g_e*(E[t]-RTE) + MFR
    FR[FR<8] = 0
    output_FR[:,t] = FR
    if any(FR):
        index = sum(1 for x in FR if x > MFR)
        for n in range(1,index):
            if FR[n] > PFR[n]:
                FR[n] = PFR[n]
            spike_train_temp = np.zeros(len(time))
            if any(spike_train[n,:]) != True:
                spike_train[n,t] = 1;
                spike_train_temp[t] = 1;
                mu = 1/float(FR[n]);
                Z = np.random.randn(1);
                if Z > 3.9:
                    Z = 3.9
                elif Z < -3.9:
                    Z = -3.9
                spike_time_temp = (mu + mu*cv*Z)*Fs;
                spike_time[n] = round(spike_time_temp) + t;
                StimulusRate = T[n]*FR[n]
                if StimulusRate > 0 and StimulusRate <= 0.4:
                    g = 1;
                elif StimulusRate > 0.4:
                    S = 1 - np.exp(-2*np.power(StimulusRate,3));
                    g = (S/float(StimulusRate))/float(0.3);
                force_temp = np.convolve(spike_train_temp,g*twitch[n,:]);
                force[n,:] = force[n,:] + force_temp[0:len(time)]

plt.plot(time,spike_train[1,:])
plt.show()
