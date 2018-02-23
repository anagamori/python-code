#!/usr/bin/env python2
# -*- coding: utf-8 -*-
"""
Created on Tue Feb 20 22:08:40 2018

@author: akiranagamori
"""



#def TwitchBasedMuscleModel():

import time
import numpy as np
import matplotlib.pyplot as plt

#    cdef double f(double x):
#        return exp(x)

def yield_function(Y,V,step):
    c_y = 0.35;
    V_y = 0.1;
    T_y = 0.2;
    Y_dot = (1-c_y*(1-np.exp(-np.abs(V)/V_y))-Y)/T_y;
    Y = Y_dot*step + Y;
    return Y 

def sag_function(S,f_eff,step):
    if f_eff < 0.1:
        a_s = 1.76;
    else:
        a_s = 0.96;
    T_s = 0.043;
    S_dot = (a_s - S)/T_s;
    S = S_dot*step + S;
    return S                

def Af_slow_function(f_eff,L,Y):
    a_f = 0.56;
    n_f0 = 2.1;
    n_f1 = 5;
    n_f = n_f0 + n_f1*(1/L-1);
    Af = 1 - np.exp(-pow(Y*f_eff/(a_f*n_f),n_f));
    return Af

def Af_fast_function(f_eff,L,S):
    a_f = 0.56;
    n_f0 = 2.1;
    n_f1 = 3.3;
    n_f = n_f0 + n_f1*(1/L-1);
    Af = 1 - np.exp(-pow(S*f_eff/(a_f*n_f),n_f));
    return Af

def Af_cor_slow_function(f_env,L,Y):
    a_f = 0.5; 
    n_f0 = 2.01;
    n_f1 = 5.16;
    n_f = n_f0 + n_f1*(1/L-1);
    FF = 1 - np.exp(-pow(Y*f_env/(a_f*n_f),n_f));
    FF = FF/f_env;
    return FF

def Af_cor_fast_function(f_env,L,Y):
    a_f = 0.52; 
    n_f0 = 1.97;
    n_f1 = 3.28;
    n_f = n_f0 + n_f1*(1/L-1);
    FF = 1 - np.exp(-pow(S*f_env/(a_f*n_f),n_f));
    FF = FF/f_env;
    return FF

def twitch_function(Af,Lce,CT,RT,Fs):
    T1 = CT*pow(Lce,2) + (CT*1/2)*Af;
    T2_temp = (RT + (RT*1/2)*Af)/Lce;
    T2 = T2_temp/1.68;
    t_twitch = np.arange(0,2,1/float(Fs));
    f_1 = np.multiply(t_twitch/T1,np.exp(1-t_twitch/T1));
    f_2 = np.multiply(t_twitch/T2,np.exp(1-t_twitch/T2));
    twitch = np.append(f_1[0:int(np.round(T1*Fs))],f_2[int(np.round(T2*Fs+1)):]);
    twitch = twitch[0:1*Fs];
    return twitch

def FL_slow_function(L):
    beta = 2.3;
    omega = 1.12;
    rho = 1.62;
    
    FL = np.exp(-pow(np.abs((pow(L,beta) - 1)/omega),rho));
    return FL

def FL_fast_function(L):
    beta = 1.55;
    omega = 0.75;
    rho = 2.12;
    
    FL = np.exp(-pow(np.abs((pow(L,beta) - 1)/omega),rho));
    return FL

def FV_con_slow_function(L,V):
    Vmax = -7.88;
    cv0 = 5.88;
    cv1 = 0;
    
    FVcon = (Vmax - V)/(Vmax + (cv0 + cv1*L)*V);
    return FVcon

def FV_con_fast_function(L,V):
    Vmax = -9.15;
    cv0 = -5.7;
    cv1 = 9.18;
    
    FVcon = (Vmax - V)/(Vmax + (cv0 + cv1*L)*V);
    return FVcon

def FV_ecc_slow_function(L,V):
    av0 = -4.7;
    av1 = 8.41;
    av2 = -5.34;
    bv = 0.35;
    FVecc = (bv - (av0 + av1*L + av2*pow(L,2))*V)/(bv+V);
    
    return FVecc

def FV_ecc_fast_function(L,V):
    av0 = -1.53;
    av1 = 0;
    av2 = 0;
    bv = 0.69;
    FVecc = (bv - (av0 + av1*L + av2*pow(L,2))*V)/(bv+V);
    
    return FVecc

def F_pe_1_function(L,V):
    c1_pe1 = 23;
    k1_pe1 = 0.046;
    Lr1_pe1 = 1.17;
    eta = 0.01;
    
    Fpe1 = c1_pe1 * k1_pe1 * np.log(np.exp((L - Lr1_pe1)/k1_pe1)+1) + eta*V;
    
    return Fpe1

def F_pe_2_function(L):
    c2_pe2 = -0.02;
    k2_pe2 = -21;
    Lr2_pe2 = 0.70;
    
    Fpe2 = c2_pe2*np.exp((k2_pe2*(L-Lr2_pe2))-1);
    return Fpe2

def F_se_function(LT):
    cT_se = 27.8; 
    kT_se = 0.0047;
    LrT_se = 0.964;
    
    Fse = cT_se * kT_se * np.log(np.exp((LT - LrT_se)/kT_se)+1);
    return Fse

def InitialLength(L0,alpha,L_slack,Lce_initial,Lt_initial):
    cT = 27.8;
    kT = 0.0047;
    LrT = 0.964;
    c1 = 23;
    k1 = 0.046;
    Lr1 = 1.17;
    PassiveForce = c1 * k1 * np.log(np.exp((1 - Lr1)/k1)+1);
    Normalized_SE_Length = kT*np.log(np.exp(PassiveForce/cT/kT)-1)+LrT;
    Lmt_temp_max = L0*np.cos(alpha)+L_slack + 1;
    L0_temp = L0;
    L0T_temp = L_slack*1.05;
    SE_Length =  L0T_temp * Normalized_SE_Length;
    FasclMax = (Lmt_temp_max - SE_Length)/L0_temp;
    Lmax = FasclMax/float(np.cos(alpha));
    Lmt_temp = Lce_initial * np.cos(alpha) + Lt_initial;
    InitialLength =  (Lmt_temp-(-L0T_temp*(kT/k1*Lr1-LrT-kT*np.log(c1/cT*k1/kT))))/(100*(1+kT/k1*L0T_temp/Lmax*1/L0_temp)*np.cos(alpha));
    Lce_initial = InitialLength/(L0_temp/100);
    Lse_initial = (Lmt_temp - InitialLength*np.cos(alpha)*100)/L0T_temp;
    
    return (Lce_initial,Lse_initial,Lmax)

    
#import matplotlib.pyplot as plt
# 
L0 = 3.0;
alpha = 10*np.pi/180;
L_slack = 3;
L0T = L_slack*1.05;
Lce_initial = 3.0;
Lt_initial = 3;
Lmt = Lce_initial*np.cos(alpha)+Lt_initial;
(Lce,Lse,Lmax) =  InitialLength(L0,alpha,L_slack,Lce_initial,Lt_initial);

density = 1.06;    
mass = 0.0001;    
PCSA = (mass*1000)/(density*L0)
sigma = 22.4;
F0 = PCSA*sigma;
Ur = 0.6;
F_pcsa_slow = 0.5;

N_MU = 120;
i = np.arange(1,N_MU+1,1);  

# Motor unit twitch force
RP_MU = 25;
b_MU = np.log(RP_MU)/float(N_MU)
P_MU = np.exp(b_MU*i)
PTi = P_MU/np.sum(P_MU)*F0
Pi_half = np.divide(PTi,2)
a_twitch = 0.0003628
b_twitch = 0.03111
Pi = a_twitch*np.exp(b_twitch*i)

# Fiber type assignment 
F_pcsa_slow = 0.5
index_slow = 96

# Recruitment threshold 
Ur = 0.6;
Ur_1 = 0.01;
a_U_th = 0.009662;
b_U_th = 0.03441;
U_th = a_U_th*np.exp(b_U_th*i); 

# Peak firing rate
MFR_MU = 8;
PFR1_MU = 20;
PFRD_MU = 30;
RTEn_MU = U_th[N_MU-1];
PFR_MU = PFR1_MU + PFRD_MU*U_th/RTEn_MU;
FR_half = PFR_MU/2

# Contraction time and half relaxation time

CT_n = 20;
FR_half_n = FR_half[N_MU-1];
CT = 1.5*np.divide(CT_n*FR_half_n,FR_half);
CT = CT - (CT[N_MU-1] - CT_n);
CT = CT/1000;
RT = CT;  

# CoV of interspike intervals 
cv = 0;


start_time = time.time()
Fs = 5000
step = 1/float(Fs)
h = step;
t_twitch = np.arange(0,1,step)
m = float(6)

amp = 0.2
time_sim = np.arange(0,5,step)
U = np.concatenate((np.zeros(1*Fs),amp/2*np.arange(0,2,step),amp*np.ones(5*Fs-3*Fs)),axis = 0)
U_eff = 0;
f_int_slow = 0;
f_eff_slow = 0;
f_eff_slow_dot = 0;
f_int_fast = 0;
f_eff_fast = 0;
f_eff_fast_dot= 0;

Vce = 0;
Y = 0;
S = 0;

MuscleVelocity = 0;
MuscleLength = Lce*L0/100;

spike_train = np.zeros((N_MU,len(time_sim)))
spike_train_temp = np.zeros(len(time_sim))
spike_time = np.zeros(N_MU)
force = np.zeros((N_MU,len(time_sim)))
Force_vec = np.zeros(len(time_sim));
ForceSE_vec = np.zeros(len(time_sim));
Lce_vec = np.zeros(len(time_sim));
U_eff_vec = np.zeros(len(time_sim));
f_env_vec = np.zeros(len(time_sim));
Y_vec = np.zeros(N_MU);
S_vec = np.zeros(N_MU);
Y_temp = float(0);
S_temp = np.zeros(N_MU,dtype=float);
Af = np.zeros(N_MU);
FF = np.zeros(N_MU);

for t in xrange(len(time_sim)): 
    if U[t] >= U_eff:
        T_U = 0.03;
    else:
        T_U = 0.15;
    
    U_eff_dot = (U[t]-U_eff)/(T_U);
    U_eff = U_eff_dot*step + U_eff;
    
    FR = np.multiply((PFR_MU-MFR_MU)/(1-U_th),(U_eff-U_th)) + MFR_MU;
    FR[FR<8] = 0;
    
    f_env = np.divide(FR,FR_half);
    force_half = np.divide(force[:,t],Pi_half);
    
    Y_temp = yield_function(Y_temp,Vce,step);
    Y_vec[0:index_slow-1] = Y_temp;
    
    for k in xrange(N_MU):
        if FR[k] > PFR_MU[k]:
            FR[k] = PFR_MU[k];
        if k <= index_slow-1:
            Af[k] = Af_slow_function(force_half[k],Lce,Y_vec[k]);
            if f_env[k] != 0:
                FF[k] = Af_cor_slow_function(f_env[k],Lce,Y_vec[k]);
        else:
            S_temp[k] = sag_function(S_temp[k],force_half[k],step);
            S_vec[k] = S_temp[k];
            Af[k] = Af_fast_function(force_half[k],Lce,S_vec[k]);
            if f_env[k] != 0:
                FF[k] = Af_cor_fast_function(f_env[k],Lce,S_vec[k]);                
  
    index_temp1 = np.all(spike_train==0,axis = 1);
    index_1 = np.where(np.logical_and(FR >= MFR_MU,index_temp1));
    index_2 = np.where(np.logical_and(FR >= MFR_MU,spike_time == t));
    index = np.append(index_1,index_2);
    
    if index.size != 0:
        for j in xrange(len(index)):
            n = index[j];
            if FR[n] > PFR_MU[n]:
                FR[n] = PFR_MU[n]   
            spike_train_temp = np.zeros(len(time_sim))
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
                twitch_temp = twitch_function(Af[n],Lce,CT[n],RT[n],Fs);
                force_temp = Pi[n]*twitch_temp*FF[n];
                if len(time_sim)-t >= len(t_twitch):                       
                    force[n,t:len(t_twitch)+t] = force[n,t:len(t_twitch)+t] + force_temp;
                else:
                    force[n,t:len(time_sim)] = force[n,t:len(time_sim)] + force_temp[0:len(time_sim)-t];    
            else:
                if spike_time[n] == t:
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
                    force_temp = Pi[n]*twitch_temp*FF[n];
                    if len(time_sim)-t >= len(t_twitch):                       
                        force[n,t:len(t_twitch)+t] = force[n,t:len(t_twitch)+t] + force_temp;
                    else:
                        force[n,t:len(time_sim)] = force[n,t:len(time_sim)] + force_temp[0:len(time_sim)-t];            
                elif t > spike_time[n] + round(1/float(FR[n])*Fs):
                    spike_train[n,t] = 1;
                    spike_train_temp[t] = 1;
                    spike_time[n] = t;                        
                    twitch_temp = twitch_function(Af[n],Lce,CT[n],RT[n],Fs);
                    force_temp = Pi[n]*twitch_temp*FF[n];                      
                    if len(time_sim)-t >= len(t_twitch):                       
                        force[n,t:len(t_twitch)+t] = force[n,t:len(t_twitch)+t] + force_temp;
                    else:
                        force[n,t:len(time_sim)] = force[n,t:len(time_sim)] + force_temp[0:len(time_sim)-t];
                if n <= index_slow:
                    FL_temp = FL_slow_function(Lce);
                    if Vce > 0:
                        FV_temp = FV_ecc_slow_function(Lce,Vce);
                    else:
                        FV_temp = FV_con_slow_function(Lce,Vce);
                else:
                    FL_temp = FL_fast_function(Lce);
                    if Vce > 0:
                        FV_temp = FV_ecc_fast_function(Lce,Vce);
                    else:
                        FV_temp = FV_con_fast_function(Lce,Vce);
                
                force[n,t] = force[n,t]*FL_temp*FV_temp;                
    
    FP1 = F_pe_1_function(Lce/Lmax,Vce);
    FP2 = F_pe_2_function(Lce);
    if FP2 > 0:
        FP2 = 0;
    
    Force = np.sum(force[:,t]) + FP1*F0 + FP2*F0                          
    
    ForceSE = F_se_function(Lse)*F0;
                 
    k_0 = h*MuscleVelocity;
    l_0 = h*((ForceSE*np.cos(alpha) - Force*np.power(np.cos(alpha),2))/mass + (np.power(MuscleVelocity,2)*np.power(np.tan(alpha),2)/MuscleLength));
    k_1 = h*(MuscleVelocity+l_0/2);
    l_1 = h*((ForceSE*np.cos(alpha) - Force*np.power(np.cos(alpha),2))/mass + (np.power(MuscleVelocity+l_0/2,2)*np.power(np.tan(alpha),2)/(MuscleLength+k_0/2)));
    k_2 = h*(MuscleVelocity+l_1/2);
    l_2 = h*((ForceSE*np.cos(alpha) - Force*np.power(np.cos(alpha),2))/mass + (np.power(MuscleVelocity+l_1/2,2)*np.power(np.tan(alpha),2)/(MuscleLength+k_1/2)));
    k_3 = h*(MuscleVelocity+l_2);
    l_3 = h*((ForceSE*np.cos(alpha) - Force*np.power(np.cos(alpha),2))/mass + (np.power(MuscleVelocity+l_2,2)*np.power(np.tan(alpha),2)/(MuscleLength+k_2)));
    MuscleLength = MuscleLength + 1/m*(k_0+2*k_1+2*k_2+k_3);
    MuscleVelocity = MuscleVelocity + 1/m*(l_0+2*l_1+2*l_2+l_3);
    
    Vce = MuscleVelocity/(L0/100);
    Lce = MuscleLength/(L0/100);
    Lse = (Lmt - Lce*L0*np.cos(alpha))/(L0T);
    
    
    ForceSE_vec[t] = ForceSE;
    Force_vec[t] = Force;
    Lce_vec[t] = Lce;
    U_eff_vec[t] = U_eff;      
    
end_time = time.time()
print(end_time - start_time)

fig1 = plt.figure()
ax1 = fig1.add_subplot(111)
ax1.plot(time_sim,ForceSE_vec)


fig2 = plt.figure()
ax2 = fig2.add_subplot(111)
ax2.plot(time_sim,Lce_vec)


#plt.show()
#plt.plot(time_sim,ForceSE_vec)
#plt.plot(time_sim,force[1,])
#plt.plot(time_sim,spike_train[1,])
#plt.show()
#return (Force_vec,ForceSE_vec);


#(Force_vec,ForceSE_vec) = TwitchBasedMuscleModel()
#(Force,ForceSE) = MuscleModel()