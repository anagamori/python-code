#!/usr/bin/env python2
# -*- coding: utf-8 -*-
"""
Created on Mon Mar 12 13:53:05 2018

@author: akira
"""

from libc.math cimport exp
from libc.math cimport cos
from libc.math cimport tan
from libc.math cimport pow
from libc.math cimport log
from libc.math cimport round
from libc.math cimport sqrt
#from cython.parallel import prange
#import multiprocessing
import time
import numpy as np
from scipy import signal
import matplotlib.pyplot as plt
import os
    

    
def TwitchBasedMuscleModel(trialN,target_amplitude,gainMatrix,control_type,noise_type,recruitment_type):

    def sign(x):
        return (x > 0) - (x < 0);
    
    def yield_function(Y,V,step):
        c_y = 0.35;
        V_y = 0.1;
        T_y = 0.2;
        Y_dot = (1-c_y*(1-exp(-abs(V)/V_y))-Y)/T_y;
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
        Af = 1 - exp(-pow(Y*f_eff/(a_f*n_f),n_f));
        return Af
    
    def Af_fast_function(f_eff,L,S):
        a_f = 0.56;
        n_f0 = 2.1;
        n_f1 = 3.3;
        n_f = n_f0 + n_f1*(1/L-1);
        Af = 1 - exp(-pow(S*f_eff/(a_f*n_f),n_f));
        return Af
    
    def Af_cor_slow_function(f_env,L,Y):
        a_f = 0.5; 
        n_f0 = 2.01;
        n_f1 = 5.16;
        n_f = n_f0 + n_f1*(1/L-1);
        FF = 1 - exp(-pow(Y*f_env/(a_f*n_f),n_f));
        FF = FF/f_env;
        return FF
    
    def Af_cor_fast_function(f_env,L,S):
        a_f = 0.52; 
        n_f0 = 1.97;
        n_f1 = 3.28;
        n_f = n_f0 + n_f1*(1/L-1);
        FF = 1 - exp(-pow(S*f_env/(a_f*n_f),n_f));
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
        
        FL = exp(-pow(abs((pow(L,beta) - 1)/omega),rho));
        return FL
    
    def FL_fast_function(L):
        beta = 1.55;
        omega = 0.75;
        rho = 2.12;
        
        FL = exp(-pow(abs((pow(L,beta) - 1)/omega),rho));
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
        
        Fpe1 = c1_pe1 * k1_pe1 * log(exp((L - Lr1_pe1)/k1_pe1)+1) + eta*V;
        
        return Fpe1
    
    def F_pe_2_function(L):
        c2_pe2 = -0.02;
        k2_pe2 = -21;
        Lr2_pe2 = 0.70;
        
        Fpe2 = c2_pe2*exp((k2_pe2*(L-Lr2_pe2))-1);
        return Fpe2
    
    def F_se_function(LT):
        cT_se = 27.8; 
        kT_se = 0.0047;
        LrT_se = 0.964;
        
        Fse = cT_se * kT_se * log(exp((LT - LrT_se)/kT_se)+1);
        return Fse
    
    def InitialLength(L0,alpha,L_slack,Lce_initial,Lt_initial):
        cT = 27.8;
        kT = 0.0047;
        LrT = 0.964;
        c1 = 23;
        k1 = 0.046;
        Lr1 = 1.17;
        PassiveForce = c1 * k1 * log(exp((1 - Lr1)/k1)+1);
        Normalized_SE_Length = kT * log(exp(PassiveForce/cT/kT)-1)+LrT;
        Lmt_temp_max = L0*cos(alpha)+L_slack + 1;
        L0_temp = L0;
        L0T_temp = L_slack*1.05;
        SE_Length =  L0T_temp * Normalized_SE_Length;
        FasclMax = (Lmt_temp_max - SE_Length)/L0_temp;
        Lmax = float(FasclMax)/cos(alpha);
        Lmt_temp = Lce_initial * cos(alpha) + Lt_initial;
        InitialLength =  (Lmt_temp-(-L0T_temp*(kT/k1*Lr1-LrT-kT*log(c1/cT*k1/kT))))/(100*(1+kT/k1*L0T_temp/Lmax*1/L0_temp)*cos(alpha));
        Lce_initial = InitialLength/(L0_temp/100);
        Lse_initial = (Lmt_temp - InitialLength*cos(alpha)*100)/L0T_temp;
        
        return (Lce_initial,Lse_initial,Lmax)
    
    def bag1_model(f_dynamic,gamma_dynamic,T,T_dot,L,V,A,step):
        p = 2;
        R = 0.46;
        a = 0.3;
        K_SR = 10.4649;
        K_PR = 0.15;
        M = 0.0002;
        LN_SR = 0.0423;
        L0_SR = 0.04; 
        L0_PR = 0.76;                 
        tau_bag1 = 0.149; 
        freq_bag1 = 60;         
        beta0 = 0.0605; 
        beta1 = 0.2592; 
        Gamma1 = 0.0289;       
        G = 20000;  
        if V >= 0:
            C = 1;
        else:
            C = 0.42;
        df_dynamic = (float(pow(gamma_dynamic,p))/(pow(gamma_dynamic,p)+pow(freq_bag1,p))-f_dynamic)/float(tau_bag1);
        f_dynamic = df_dynamic*step + f_dynamic;
        
        beta = beta0 + beta1 * f_dynamic; 
        Gamma = Gamma1 * f_dynamic;
        
        T_ddot = K_SR/M * ((C*beta*sign(V-T_dot/K_SR)*pow(abs(V-T_dot/K_SR),a))*(L-L0_SR-T/K_SR-R)+K_PR*(L-L0_SR-T/K_SR-L0_PR)+M*A+Gamma-T);
        T_dot = T_ddot*step + T_dot;
        T = T_dot*step + T;
               
        AP_bag1 = G*(T/K_SR-(LN_SR-L0_SR));
        
        return (AP_bag1,f_dynamic,T,T_dot)
    
    def bag2_model(f_static,gamma_static,T,T_dot,L,V,A,step):
        p = 2;
        R = 0.46;
        a = 0.3;
        K_SR = 10.4649;
        K_PR = 0.15;
        M = 0.0002;        
        LN_SR = 0.0423;
        LN_PR = 0.89;
        L0_SR = 0.04; 
        L0_PR = 0.76;      
        L_secondary = 0.04;
        X = 0.7;
        tau_bag2 = 0.205; 
        freq_bag2 = 60;         
        beta0 = 0.0822; 
        beta2 = -0.046; 
        Gamma2 = 0.0636;       
        G = 10000;  
        if V >= 0:
            C = 1;
        else:
            C = 0.42;
        df_static = (float(pow(gamma_static,p))/(pow(gamma_static,p)+pow(freq_bag2,p))-f_static)/float(tau_bag2);
        f_static = df_static*step + f_static;
        beta = beta0 + beta2 * f_static;
        Gamma = Gamma2 * f_static;
        
        T_ddot = K_SR/M * ((C*beta*sign(V-T_dot/K_SR)*pow(abs(V-T_dot/K_SR),a))*(L-L0_SR-T/K_SR-R)+K_PR*(L-L0_SR-T/K_SR-L0_PR)+M*A+Gamma-T);
        T_dot = T_ddot*step + T_dot;
        T = T_dot*step + T;
               
        AP_primary_bag2 = G*(T/K_SR-(LN_SR-L0_SR));
        AP_secondary_bag2 = G*(X*L_secondary/L0_SR*(T/K_SR-(LN_SR-L0_SR))+(1-X)*L_secondary/L0_PR*(L-T/K_SR-L0_SR-LN_PR))
                
        return (AP_primary_bag2,AP_secondary_bag2,f_static,T,T_dot)
    
    def chain_model(gamma_static,T,T_dot,L,V,A,step):
        p = 2;
        R = 0.46;
        a = 0.3;
        K_SR = 10.4649;
        K_PR = 0.15;
        M = 0.0002;        
        LN_SR = 0.0423;
        LN_PR = 0.89;
        L0_SR = 0.04; 
        L0_PR = 0.76;      
        L_secondary = 0.04;
        X = 0.7;
        freq_chain = 90;
        beta0 = 0.0822;
        beta2_chain = -0.069;
        Gamma2_chain = 0.0954;
        G = 10000;
        if V >= 0:
            C = 1;
        else:
            C = 0.42;
        
        f_static_chain = float(pow(gamma_static,p))/(pow(gamma_static,p)+pow(freq_chain,p));
        beta = beta0 + beta2_chain * f_static_chain;
        Gamma = Gamma2_chain * f_static_chain;
        
        T_ddot = K_SR/M * ((C*beta*sign(V-T_dot/K_SR)*pow(abs(V-T_dot/K_SR),a))*(L-L0_SR-T/K_SR-R)+K_PR*(L-L0_SR-T/K_SR-L0_PR)+M*A+Gamma-T);
        T_dot = T_ddot*step + T_dot;
        T = T_dot*step + T;
        
        AP_primary_chain = G*(T/K_SR-(LN_SR-L0_SR));
        AP_secondary_chain = G*(X*L_secondary/L0_SR*(T/K_SR-(LN_SR-L0_SR))+(1-X)*L_secondary/L0_PR*(L-T/K_SR-L0_SR-LN_PR))
                  
        return (AP_primary_chain,AP_secondary_chain,T,T_dot)
    
    def SpindleOutput(AP_bag1,AP_primary_bag2,AP_secondary_bag2,AP_primary_chain,AP_secondary_chain):
        S = 0.156;
        
        if AP_bag1 < 0:
            AP_bag1 = 0;
        if AP_primary_bag2 < 0:
            AP_primary_bag2 = 0;
        if AP_primary_chain < 0:
            AP_primary_chain = 0;
        if AP_secondary_bag2 < 0:
            AP_secondary_bag2 = 0;
        if AP_secondary_chain < 0:
            AP_secondary_chain = 0;
        if AP_bag1 > (AP_primary_bag2+AP_primary_chain):
            Larger = AP_bag1;
            Smaller = AP_primary_bag2+AP_primary_chain;    
        else:
            Larger = AP_primary_bag2+AP_primary_chain;
            Smaller = AP_bag1;
            
        Output_Primary = Larger + S * Smaller;
        Output_Secondary = AP_secondary_bag2 + AP_secondary_chain;
        if Output_Primary < 0:
            Output_Primary = 0;
        elif Output_Primary > 100000:
            Output_Primary = 100000;        
        if Output_Secondary < 0:
            Output_Secondary = 0;
        elif Output_Secondary > 100000:
            Output_Secondary = 100000;
            
        return (Output_Primary,Output_Secondary)
    
    def GTOOutput(FR_Ib,FR_Ib_temp,x_GTO,Force,index):
        cdef double G1 = 60;
        cdef double G2 = 4;
        cdef double num1 = 1.7;
        cdef double num2 = -3.399742022978487;
        cdef double num3 = 1.699742026978047;
        cdef double den1 = 1.0;
        cdef double den2 = -1.999780020198665;
        cdef double den3 = 0.999780024198225;
                
        x_GTO[index] = G1*log(Force/G2+1);
              
        FR_Ib_temp[index] = (num3*x_GTO[index-2] + num2*x_GTO[index-1] + num1*x_GTO[index] - 
                  den3*FR_Ib_temp[index-2] - den2*FR_Ib_temp[index-1])/den1;
        FR_Ib[index] = FR_Ib_temp[index];
        if FR_Ib[index]<0:
            FR_Ib[index] = 0;
        
        return (FR_Ib,FR_Ib_temp,x_GTO)
    
    def RenshawOutput(FR_RI,FR_RI_temp,ND,index):
        cdef double num1 = 0.238563173450928;
        cdef double num2 = -0.035326319453965;
        cdef double num3 = -0.200104635331441;
        cdef double den1 = 1.0;
        cdef double den2 = -1.705481699867712;
        cdef double den3 = 0.708613918533233;
        
        FR_RI_temp[index] = (num3*ND[-1-2]+num2*ND[-1-1]+num1*ND[-1]-den3*FR_RI_temp[index-2]-den2*FR_RI_temp[index-1])/den1;
        FR_RI[index] = FR_RI_temp[index]; 
        if FR_RI[index] < 0:
            FR_RI[index] = 0;
        
        return (FR_RI,FR_RI_temp)
    
    def smoothSaturationFunction(x):
        y = 1/float((1+exp(-11*(x-0.5))));
        return y
    
    def actionPotentialGeneration(exc_input,inh_input,n_exc,n_inh,IC):
        cdef double HYP = 2.0;
        cdef double OD = 2.0;        
        s_inh = - HYP/n_inh;
        s_exc = (1 + OD)/n_exc;
        y = s_exc*(exc_input) + s_inh*(inh_input) + IC;
        return y 
    
    def noiseOutput(noise,noise_filt,Input,index):
        cdef double noise_amp = 0.0005;
        cdef double b1 = 0.089848614641397*1e-5;
        cdef double b2 = 0.359394458565587*1e-5;
        cdef double b3 = 0.539091687848381*1e-5;
        cdef double b4 = 0.359394458565587*1e-5;
        cdef double b5 = 0.089848614641397*1e-5;
        cdef double a1 = 1.0;
        cdef double a2 = -3.835825540647348;
        cdef double a3 = 5.520819136622229;
        cdef double a4 = -3.533535219463015;
        cdef double a5 = 0.848555999266477;
        r = np.random.rand(1);
        noise[index] = 2*(r-0.5)*(sqrt(noise_amp*Input)*sqrt(3));
        noise_filt[index] = (b5*noise[index-4] + b4*noise[index-3] + b3*noise[index-2] + b2*noise[index-1] +
            b1*noise[index] - a5*noise_filt[index-4] - a4*noise_filt[index-3] - a3*noise_filt[index-2] -
            a2*noise_filt[index-1])/a1;
        return (noise,noise_filt)
        
    #import matplotlib.pyplot as plt
    # 
    cdef double Lse
    cdef double Lce
    cdef double Lmax
    cdef double L0 = 5.1;
    cdef double alpha = 3.1*np.pi/180;
    cdef double L_slack = 27.1;
    cdef double L0T = L_slack*1.05;
    cdef double Lce_initial = 5.1;
    cdef double Lt_initial = 27.1;
    cdef double Lmt = Lce_initial*cos(alpha)+Lt_initial;
    (Lce,Lse,Lmax) =  InitialLength(L0,alpha,L_slack,Lce_initial,Lt_initial);
    
    cdef double density = 1.06;    
    cdef double mass = 0.02;    
    PCSA = (mass*1000)/(density*L0)
    sigma = 31.8;
    F0 = PCSA*sigma;
    Ur = 0.6;
    F_pcsa_slow = 0.5;
    cdef double F_max = 74.317;
    
    cdef int N_MU = 300;
    i_MU = np.arange(1,N_MU+1,1);  
    
    # Motor unit twitch force
    RP_MU = 25;
    b_MU = log(RP_MU)/float(N_MU)
    P_MU = np.exp(b_MU*i_MU)
    PTi = P_MU/np.sum(P_MU)*F0
    Pi_half = np.divide(PTi,2)
    cdef double a_twitch = 0.021927760103178
    cdef double b_twitch = 0.012199419376977
    Pi = a_twitch*np.exp(b_twitch*i_MU)
    
    # Fiber type assignment 
    F_pcsa_slow = 0.5
    index_slow = 239
    
    # Recruitment threshold 
    Ur = 0.6;
    Ur_1 = 0.01;
    a_U_th = 0.009863998688657;
    b_U_th = 0.013693460074321;
    U_th = a_U_th*np.exp(b_U_th*i_MU); 
    
    # Peak firing rate
    MFR1_MU = 8;
    MFRn_MU = 14;
    RTEn_MU = U_th[N_MU-1]-Ur_1;
    MFR_MU = MFR1_MU + (MFRn_MU-MFR1_MU)*((U_th-Ur_1)/RTEn_MU);
    PFR_MU = 4*MFR_MU;    
    FR_half = PFR_MU/2
    g_e = (PFR_MU[-1] - MFR_MU[-1])/(1-U_th[-1]);
    
    # Contraction time and half relaxation time
    CT_n = 20;
    FR_half_n = FR_half[N_MU-1];
    CT = 1.5*np.divide(CT_n*FR_half_n,FR_half);
    CT = CT - (CT[N_MU-1] - CT_n);
    CT = CT/1000;
    RT = CT;
    
    # CoV of interspike intervals 
    cv = 0.1;
    
    # Simulation     
    start_time = time.time()
    Fs = 10000
    step = 1/float(Fs)
    h = step;
    t_twitch = np.arange(0,1,step)
    cdef double m = 6
    
    cdef double amp = target_amplitude;
    duration = 10;
    time_sim = np.arange(0,duration,step)
    Input = np.concatenate((np.zeros(1*Fs),amp/2*np.arange(0,2,step),amp*np.ones(duration*Fs-3*Fs)),axis = 0)
    F_target = F_max*Input;      
    # Spindle 
    cdef double f_dynamic_bag1 = 0;
    cdef double T_bag1 = 0;
    cdef double T_dot_bag1 = 0;
    cdef double f_static_bag2 = 0;
    cdef double T_bag2 = 0;
    cdef double T_dot_bag2 = 0;
    cdef double T_chain = 0;
    cdef double T_dot_chain = 0;    
    
    
    # Delay parameters
    cdef double distance_Muscle2SpinalCord = 0.8;
    cdef double conductionVelocity_efferent = 48.5;
    cdef double conductionVelocity_Ia = 64.5;
    cdef double conductionVelocity_Ib = 59.0;
    cdef double synaptic_delay = 2.0;
    
    delay_efferent = int(round(distance_Muscle2SpinalCord/conductionVelocity_efferent*1000) + synaptic_delay)*Fs/1000;
    delay_Ia = int(round(distance_Muscle2SpinalCord/conductionVelocity_Ia*1000) + synaptic_delay)*Fs/1000;
    delay_Ib = int(round(distance_Muscle2SpinalCord/conductionVelocity_Ib*1000) + synaptic_delay)*Fs/1000;
    cdef int delay_C = 50*Fs/1000;
    cdef int delay_synaptic = 2;
    
    # Gain parameters
    cdef double K = 0.0015;
    cdef double Gain_Ia = 400.0;
    cdef double Gain_Ib = 400.0;
    cdef double Gain_RI = 3.0;
    cdef double gamma_dynamic = gainMatrix[0];
    cdef double gamma_static = gainMatrix[1];
    cdef double Ia_PC = gainMatrix[2];
    cdef double Ib_PC = gainMatrix[3];
    cdef double RI_PC = gainMatrix[4];
    cdef double PN_PC_Ia = gainMatrix[5];
    cdef double PN_PC_Ib = gainMatrix[6];
    cdef double PN_PC = gainMatrix[7];
    
    # Muscle length 
    cdef double MuscleVelocity = 0;
    cdef double MuscleLength = Lce*L0/100;
    
    # Parameter initialization
    # Neural drive
    cdef double U_eff = 0.0;  
    cdef double Input_C_temp = 0.0;
    cdef double Input_C = 0.0;
    cdef double ND_temp = 0.0;
    # Muscle dynamics  
    Ace = 0;
    Vce = 0;   
    Y = 0;
    S = 0;
    
    spike_train = np.zeros((N_MU,len(time_sim)))
    spike_time = np.zeros(N_MU)
    force = np.zeros((N_MU,len(time_sim)))
    Force_vec = np.zeros(len(time_sim));
    ForceSE_vec = np.zeros(len(time_sim));
    Lce_vec = np.zeros(len(time_sim));
    Vce_vec = np.zeros(len(time_sim));
    Ace_vec = np.zeros(len(time_sim));
    FL_slow_vec = np.zeros(len(time_sim));
    FL_fast_vec = np.zeros(len(time_sim));
    FV_slow_vec = np.zeros(len(time_sim));
    FV_fast_vec = np.zeros(len(time_sim));
    U_eff_vec = np.zeros(len(time_sim));
    f_env_vec = np.zeros(len(time_sim));
    FR =  np.zeros(N_MU);
    f_env =  np.zeros(N_MU);
    force_half =  np.zeros(N_MU);
    Y_vec = np.zeros(N_MU);
    S_vec = np.zeros(N_MU);
    Y_temp = float(0);
    S_temp = np.zeros(N_MU);
    Y_vec_store = np.zeros(len(time_sim));
    S_vec_store = np.zeros(len(time_sim));
    Af = np.zeros(N_MU);
    FF = np.zeros(N_MU);
    FR_mat = np.zeros((N_MU,len(time_sim)));
    f_env_mat = np.zeros((N_MU,len(time_sim)));
    force_half_mat = np.zeros((N_MU,len(time_sim)));    
    Af_mat = np.zeros((N_MU,len(time_sim)));
    FF_mat = np.zeros((N_MU,len(time_sim)));
    cdef double ForceSE = F_se_function(Lse) * F0;
    
    
    Ia_vec = np.zeros(len(time_sim));
    FR_Ia = np.zeros(len(time_sim));
    Input_Ia = np.zeros(len(time_sim));
    
    x_GTO = np.zeros(len(time_sim));
    FR_Ib_temp = np.zeros(len(time_sim));
    FR_Ib = np.zeros(len(time_sim));
    Ib_vec = np.zeros(len(time_sim));
    Input_Ib = np.zeros(len(time_sim));
    
    FR_RI_temp = np.zeros(len(time_sim));
    FR_RI = np.zeros(len(time_sim));
    RI_vec = np.zeros(len(time_sim));
    Input_RI = np.zeros(len(time_sim));
    
    FR_PN = np.zeros(len(time_sim));
    PN_vec = np.zeros(len(time_sim));
    Input_PN = np.zeros(len(time_sim));
    
    # noise related
    # noise for neural drive
    noise = np.zeros(len(time_sim));
    noise_filt = np.zeros(len(time_sim));
    # noise for error-based controller
    noise_C = np.zeros(len(time_sim));
    noise_C_filt = np.zeros(len(time_sim));
    # noise for Ia
    noise_Ia = np.zeros(len(time_sim));
    noise_Ia_filt = np.zeros(len(time_sim));
    # noise for Ib
    noise_Ib = np.zeros(len(time_sim));
    noise_Ib_filt = np.zeros(len(time_sim));
    # noise for RI
    noise_RI = np.zeros(len(time_sim));
    noise_RI_filt = np.zeros(len(time_sim));
    # noise for PN
    noise_PN = np.zeros(len(time_sim));
    noise_PN_filt = np.zeros(len(time_sim));
    
    C_vec = np.zeros(len(time_sim));
    ND = np.zeros(len(time_sim));
    ND_delayed = np.zeros(len(time_sim));
    
    for t in xrange(len(time_sim)):
           
        # Obtain spindle output
        (AP_bag1,f_dynamic_bag1,T_bag1,T_dot_bag1) = bag1_model(f_dynamic_bag1,gamma_dynamic,T_bag1,T_dot_bag1,Lce,Vce,Ace,step);        
        (AP_primary_bag2,AP_secondary_bag2,f_static,T_bag2,T_dot_bag2) = bag2_model(f_static_bag2,gamma_static,T_bag2,T_dot_bag2,Lce,Vce,Ace,step); 
        (AP_primary_chain,AP_secondary_chain,T_chain,T_dot_chain) = chain_model(gamma_static,T_chain,T_dot_chain,Lce,Vce,Ace,step);
        (Output_Primary,Output_Secondary) = SpindleOutput(AP_bag1,AP_primary_bag2,AP_secondary_bag2,AP_primary_chain,AP_secondary_chain);
        FR_Ia[t] = Output_Primary;
        #Input_Ia_temp = smoothSaturationFunction(FR_Ia[t]/Gain_Ia + Ia_PC);
        Input_Ia_temp = FR_Ia[t]/Gain_Ia + Ia_PC;
        if t > 5:
            (noise_Ia,noise_Ia_filt) = noiseOutput(noise_Ia,noise_Ia_filt,abs(Input_Ia_temp),t); 
            Input_Ia[t] = Input_Ia_temp + noise_Ia_filt[t];
            if Input_Ia[t] < 0:
                Input_Ia[t] = 0;
        
        # Obtain GTO output
        if t > 5:
            (FR_Ib,FR_Ib_temp,x_GTO) = GTOOutput(FR_Ib,FR_Ib_temp,x_GTO,ForceSE,t);
            #Input_Ib_temp = smoothSaturationFunction(FR_Ib[t]/Gain_Ib + Ib_PC);
            Input_Ib_temp = FR_Ib[t]/Gain_Ib + Ib_PC;
            (noise_Ib,noise_Ib_filt) = noiseOutput(noise_Ib,noise_Ib_filt,abs(Input_Ib_temp),t); 
            Input_Ib[t] = Input_Ib_temp + noise_Ib_filt[t];
            if Input_Ib[t] < 0:
                Input_Ib[t] = 0;
                           
        # Obtain Renshaw cell output
        if t > 5:
            (FR_RI,FR_RI_temp) = RenshawOutput(FR_RI,FR_RI_temp,ND[:t],t);
            #Input_RI_temp = smoothSaturationFunction(FR_RI[t]/Gain_RI + RI_PC);
            Input_RI_temp = FR_RI[t]/Gain_RI + RI_PC;
            (noise_RI,noise_RI_filt) = noiseOutput(noise_RI,noise_RI_filt,abs(Input_RI_temp),t); 
            Input_RI[t] = Input_RI_temp + noise_RI_filt[t];
            if Input_RI[t] < 0:
                Input_RI[t] = 0;
        
        # Obtain propriospinal neuron output
        if t > delay_Ib:
            FR_PN[t] = smoothSaturationFunction(FR_Ia[t-delay_Ia]/Gain_Ia + PN_PC_Ia)+smoothSaturationFunction(FR_Ib[t-delay_Ib]/Gain_Ib + PN_PC_Ib);
            #Input_PN_temp = smoothSaturationFunction(FR_PN[t] + PN_PC);
            Input_PN_temp = FR_PN[t] + PN_PC;
            (noise_PN,noise_PN_filt) = noiseOutput(noise_PN,noise_PN_filt,abs(Input_PN_temp),t); 
            Input_PN[t] = Input_PN_temp + noise_PN_filt[t];
            if Input_PN[t] < 0:
                Input_PN[t] = 0;
            
        
        # Integreate all the inputs
        if t > delay_C:
            Input_C_temp += K*(F_target[t]-ForceSE_vec[t-delay_C])/F_max;
            (noise_C,noise_C_filt) = noiseOutput(noise_C,noise_C_filt,abs(Input_C_temp),t); 
            Input_C = Input_C_temp + noise_C_filt[t];
            
            # add all excitatory signals
            exc_input = Input_Ia[t-delay_Ia] + Input_PN[t-delay_synaptic];
            # add all inhibitory signals
            inh_input = Input_Ib[t-delay_Ib] + Input_RI[t-delay_synaptic];  
            
            if control_type == 0: # feedforward control
                ND_temp = actionPotentialGeneration(exc_input,inh_input,2,2,Input[t]);
            else: # feedback control
                ND_temp = actionPotentialGeneration(exc_input,inh_input,2,2,Input_C);
            if ND_temp < 0:
                ND_temp = 0;
            
            if noise_type == 0:
                (noise,noise_filt) = noiseOutput(noise,noise_filt,0.1,t); 
            else:
                (noise,noise_filt) = noiseOutput(noise,noise_filt,ND_temp,t); 
            
        # Calculate neural drive to motor unit pool
        ND[t] = ND_temp + noise_filt[t];
        
        if ND[t] < 0:
            ND[t] = 0;
                
        if t > delay_efferent:
            ND_delayed[t] = ND[t-delay_efferent];      
       
        U = ND_delayed[t];
        
        if U >= U_eff:
            T_U = 0.03;
        else:
            T_U = 0.15;
        
        U_eff_dot = (U-U_eff)/(T_U);
        U_eff = U_eff_dot*step + U_eff;
        
        Y_temp = yield_function(Y_temp,Vce,step);
        Y_vec[0:index_slow-1] = Y_temp;
        
        for k in xrange(N_MU):
            if recruitment_type == 0:
                FR[k] = (PFR_MU[k]-MFR_MU[k])/(1-U_th[k])*(U_eff-U_th[k]) + MFR_MU[k];                
            elif recruitment_type == 1:
                FR[k] = g_e*(U_eff-U_th[k]) + MFR_MU[k];
            if FR[k] < MFR_MU[k]:
                FR[k] = 0;    
            if FR[k] > PFR_MU[k]:
                FR[k] = PFR_MU[k];
            FR_mat[k,t] = FR[k];
            f_env[k] = FR[k]/FR_half[k];
            force_half[k] = force[k,t]/Pi_half[k];
            if k <= index_slow-1:
                Af[k] = Af_slow_function(f_env[k],Lce,Y_vec[k]);
                if f_env[k] != 0:
                    FF[k] = Af_cor_slow_function(f_env[k],Lce,Y_vec[k]);
            else:
                S_temp[k] = sag_function(S_temp[k],f_env[k],step);
                S_vec[k] = S_temp[k];
                Af[k] = Af_fast_function(f_env[k],Lce,S_vec[k]);
                if f_env[k] != 0:
                    FF[k] = Af_cor_fast_function(f_env[k],Lce,S_vec[k]);                
         
        if t > 1:               
            index_1 = np.where(np.logical_and(FR > 0,FR_mat[:,t-1] == 0));
            index_2 = np.where(np.logical_and(FR > 0,spike_time == t));
            index = np.append(index_1,index_2);
            
            if index.size != 0:
                for j in xrange(len(index)):
                    n = index[j];                
                    if any(spike_train[n,:]) != True:
                        spike_train[n,t] = 1;                    
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
                        elif t > spike_time[n] + round(1/float(FR[n])*Fs):
                            spike_train[n,t] = 1;                        
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
                    
                            
        FL_slow = FL_slow_function(Lce);
        if Vce > 0:
            FV_slow = FV_ecc_slow_function(Lce,Vce);
        else:
            FV_slow = FV_con_slow_function(Lce,Vce);
        FL_fast = FL_fast_function(Lce);
        if Vce > 0:
            FV_fast = FV_ecc_fast_function(Lce,Vce);
        else:
            FV_fast = FV_con_fast_function(Lce,Vce);
        force[:index_slow,t] = force[:index_slow,t]*FL_slow*FV_slow;    
        force[index_slow+1:,t] = force[index_slow+1:,t]*FL_fast*FV_fast;    
                        
        FP1 = F_pe_1_function(Lce/Lmax,Vce);
        FP2 = F_pe_2_function(Lce);
        if FP2 > 0:
            FP2 = 0;
        
        Force = np.sum(force[:,t]) + FP1*F0 + FP2*F0                          
        
        ForceSE = F_se_function(Lse)*F0;
        
        MuscleAcceleration = (ForceSE*cos(alpha) - Force*pow(cos(alpha),2))/mass + (pow(MuscleVelocity,2)*pow(tan(alpha),2)/MuscleLength);    
        k_0 = h*MuscleVelocity;
        l_0 = h*((ForceSE*cos(alpha) - Force*pow(cos(alpha),2))/mass + (pow(MuscleVelocity,2)*pow(tan(alpha),2)/MuscleLength));
        k_1 = h*(MuscleVelocity+l_0/2);
        l_1 = h*((ForceSE*cos(alpha) - Force*pow(cos(alpha),2))/mass + (pow(MuscleVelocity+l_0/2,2)*pow(tan(alpha),2)/(MuscleLength+k_0/2)));
        k_2 = h*(MuscleVelocity+l_1/2);
        l_2 = h*((ForceSE*cos(alpha) - Force*pow(cos(alpha),2))/mass + (pow(MuscleVelocity+l_1/2,2)*pow(tan(alpha),2)/(MuscleLength+k_1/2)));
        k_3 = h*(MuscleVelocity+l_2);
        l_3 = h*((ForceSE*cos(alpha) - Force*pow(cos(alpha),2))/mass + (pow(MuscleVelocity+l_2,2)*pow(tan(alpha),2)/(MuscleLength+k_2)));
        MuscleLength = MuscleLength + 1/m*(k_0+2*k_1+2*k_2+k_3);
        MuscleVelocity = MuscleVelocity + 1/m*(l_0+2*l_1+2*l_2+l_3);
        
        Ace = MuscleAcceleration/(L0/100);
        Vce = MuscleVelocity/(L0/100);
        Lce = MuscleLength/(L0/100);
        Lse = (Lmt - Lce*L0*cos(alpha))/(L0T);
        
        
        ForceSE_vec[t] = ForceSE;
        Force_vec[t] = Force;
        Lce_vec[t] = Lce;
        Vce_vec[t] = Vce;
        Ace_vec[t] = Ace;
        
        #FR_mat[:,t] = FR;
        f_env_mat[:,t] = f_env;
        force_half_mat[:,t] = force_half;
        Y_vec_store[t] = Y_temp;
        S_vec_store[t] = S_vec[240];
        Af_mat[:,t] = Af;
        FF_mat[:,t] = FF;
        FL_slow_vec[t] = FL_slow;
        FL_fast_vec[t] = FL_fast;
        FV_slow_vec[t] = FV_slow;
        FV_fast_vec[t] = FV_fast;
        
        U_eff_vec[t] = U_eff; 
        
        Ia_vec[t] = FR_Ia[t];        
        
        Ib_vec[t] = FR_Ib[t];
        
        RI_vec[t] = FR_RI[t];
        
        PN_vec[t] = FR_PN[t];
        
        C_vec[t] = Input_C;
        
        
    end_time = time.time()
    print(end_time - start_time)
    
    output = {'Time':time_sim,'Tendon Force':ForceSE_vec, 'Muscle Force':Force_vec, 
              'Twitch Force': force, 'Spike Train':spike_train, 'Muscle Length': Lce_vec,
              'Muscle Velocity': Vce_vec, 'Muscle Acceleration': Ace_vec,
              'Y': Y_vec_store, 'S': S_vec_store,
              'Af': Af_mat, 'FF': FF_mat,
              'FL_slow': FL_slow_vec, 'FV_slow': FV_slow_vec,
              'FL_fast': FL_fast_vec, 'FV_fast': FV_fast_vec,
              'Ia': Ia_vec, 'Input Ia' : Input_Ia,
              'Ib': Ib_vec, 'Input Ib' : Input_Ib,
              'RI': RI_vec, 'Input RI' : Input_RI,
              'PN': PN_vec, 'Input PN': Input_PN,             
              'C': C_vec, 'U_eff': U_eff_vec,
              'FR': FR_mat}             
    
    
    default_path = '/Users/akira/Documents/Github/python-code/';  
    save_path = '/Volumes/DATA2/SDN_Data'; 
    fileName = "%s%s%s" % ('output_SDN_singleTrial_0_1_0_0_',str(trialN),'.npy')           
    os.chdir(save_path)
    np.save(fileName,output)
    os.chdir(default_path)
    
#    fig1 = plt.figure()
#    ax1 = fig1.add_subplot(111)
#    ax1.plot(time_sim,ForceSE_vec)
      
    Force_plot = ForceSE_vec[7*Fs:];
    CoV = np.std(Force_plot)/np.mean(Force_plot);
    print (CoV)
#    f,Pxx = signal.periodogram(Force_plot-np.mean(Force_plot),Fs);
#    fig2 = plt.figure(2)
#    ax2 = fig2.add_subplot(111);
#    ax2.plot(f,Pxx);
#    ax2.set_xlim([0,30]);
  
#    plt.show()
    #plt.plot(time_sim,ForceSE_vec)
    return (Force_vec,ForceSE_vec);


target_amplitude_vec = np.arange(0.1,1.1,0.1)
#target_amplitude_vec = [0.7,0.8,0.9,1.0];
gain_temp_gamma = [10,30,50,70,90,110,130];
gain_temp = [-0.5,-0.4,-0.3,-0.2,-0.1,0,0.1];
gainMatrix = [10,10,-0.5,-0.5,-0.5,-0.5,-0.5,-0.5];
control_type = 0; # 0: feedforward control, 1 feedback control
noise_type = 1; # 0: constant additive noise, 1 signal dependent additive noise
recruitment_type = 0;
#i = len(target_amplitude_vec);    #i = 3;
target_amplitude = 0.1; #target_amplitude_vec[i];
trialN = 0;    
(Force_vec,ForceSE_vec) = TwitchBasedMuscleModel(trialN,target_amplitude,gainMatrix,control_type,noise_type,recruitment_type)
#(Force,ForceSE) = MuscleModel()