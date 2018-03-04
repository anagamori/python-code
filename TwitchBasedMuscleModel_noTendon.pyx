#!/usr/bin/env python2
# -*- coding: utf-8 -*-
"""
Created on Sat Mar  3 21:51:33 2018

@author: akiranagamori
"""
from libc.math cimport exp
from libc.math cimport cos
from libc.math cimport tan
from libc.math cimport pow
from libc.math cimport log
from libc.math cimport round
from libc.math cimport sqrt

def TwitchBasedMuscleModel():

    import time
    import numpy as np
    import matplotlib.pyplot as plt
    import os
    
#    cdef double f(double x):
#        return exp(x)
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
        T1 = CT*pow(Lce,2) + (CT/2)*Af;
        T2_temp = (RT + (RT/2)*Af)/Lce;
        T2 = T2_temp/1.68;
        t_twitch = np.arange(0,2,1/float(Fs));
        f_1 = np.multiply(t_twitch/T1,np.exp(1-t_twitch/T1));
        f_2 = np.multiply(t_twitch/T2,np.exp(1-t_twitch/T2));
        twitch = np.append(f_1[0:int(np.round(T1*Fs+1))],f_2[int(np.round(T2*Fs+1)):]);
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
        
        
    #import matplotlib.pyplot as plt
    # 
    cdef double Lse
    cdef double Lce = 1.0
    cdef double Lmax 
    cdef double L0 = 3.0;
    cdef double alpha = 10*np.pi/180;
    cdef double L_slack = 0.5;
    cdef double L0T = L_slack*1.05;
    cdef double Lce_initial = 3.0;
    cdef double Lt_initial = 0.5;
    cdef double Lmt = Lce_initial*cos(alpha)+Lt_initial;
    (Lce,Lse,Lmax) =  InitialLength(L0,alpha,L_slack,Lce_initial,Lt_initial);
    
    cdef double density = 1.06;    
    cdef double mass = 0.0001;    
    PCSA = (mass*1000)/(density*L0)
    sigma = 22.4;
    F0 = PCSA*sigma;
    Ur = 0.6;
    F_pcsa_slow = 0.5;
    
    cdef int N_MU = 120;
    i_MU = np.arange(1,N_MU+1,1);  
    
    # Motor unit twitch force
    RP_MU = 25;
    b_MU = log(RP_MU)/float(N_MU)
    P_MU = np.exp(b_MU*i_MU)
    PTi = P_MU/np.sum(P_MU)*F0
    Pi_half = np.divide(PTi,2)
    cdef double a_twitch = 0.000334642434591
    cdef double b_twitch = 0.031078513781294
    Pi = a_twitch*np.exp(b_twitch*i_MU)
    
    # Fiber type assignment 
    F_pcsa_slow = 0.5
    index_slow = 96
    
    # Recruitment threshold 
    Ur = 0.6;
    Ur_1 = 0.01;
    a_U_th = 0.009661789081194;
    b_U_th = 0.034406256825396;
    U_th = a_U_th*np.exp(b_U_th*i_MU); 
    
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
    
    # Simulation     
    start_time = time.time()
    Fs = 1000
    step = 1/float(Fs)
    h = step;
    t_twitch = np.arange(0,1,step)
    cdef double m = 6
    
    cdef double amp = 1.0
    time_sim = np.arange(0,5,step)
    Input = np.concatenate((np.zeros(1*Fs),amp/2*np.arange(0,2,step),amp*np.ones(5*Fs-3*Fs)),axis = 0)
    F_target = F0*Input;      

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
    U_eff_vec = np.zeros(len(time_sim));
    f_env_vec = np.zeros(len(time_sim));
    Af_vec = np.zeros(len(time_sim));
    FF_vec = np.zeros(len(time_sim));
    S_vec_store = np.zeros(len(time_sim));
    Y_vec = np.zeros(N_MU);
    S_vec = np.zeros(N_MU);
    Y_temp = float(0);
    S_temp = 1.76*np.ones(N_MU);
    Af = np.zeros(N_MU);
    FF = np.zeros(N_MU);
    cdef double ForceSE = F_se_function(Lse) * F0;   
    
    for t in xrange(len(time_sim)):           
       
        U = Input[t];
        
        if U >= U_eff:
            T_U = 0.03;
        else:
            T_U = 0.15;
        
        U_eff_dot = (U-U_eff)/(T_U);
        U_eff = U_eff_dot*step + U_eff;
        
        
        # Calculate firing rate of each unit
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
                S_temp[k] = sag_function(S_temp[k],f_env[k],step);
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
                    twitch = Pi[n]*twitch_temp*FF[n]; 
                    if len(time_sim)-t >= len(t_twitch):                       
                        force[n,t:len(t_twitch)+t] = force[n,t:len(t_twitch)+t] + twitch;
                    else:
                        force[n,t:len(time_sim)] = force[n,t:len(time_sim)] + twitch[0:len(time_sim)-t];    
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
                        twitch = Pi[n]*twitch_temp*FF[n];                        
                        if len(time_sim)-t >= len(t_twitch):                       
                            force[n,t:len(t_twitch)+t] = force[n,t:len(t_twitch)+t] + twitch;
                        else:
                            force[n,t:len(time_sim)] = force[n,t:len(time_sim)] + twitch[0:len(time_sim)-t];            
                    elif t > spike_time[n] + round(1/float(FR[n])*Fs):
                        spike_train[n,t] = 1;                        
                        spike_time[n] = t;                        
                        twitch_temp = twitch_function(Af[n],Lce,CT[n],RT[n],Fs);
                        twitch = Pi[n]*twitch_temp*FF[n]; 
                        if len(time_sim)-t >= len(t_twitch):                       
                            force[n,t:len(t_twitch)+t] = force[n,t:len(t_twitch)+t] + twitch;
                        else:
                            force[n,t:len(time_sim)] = force[n,t:len(time_sim)] + twitch[0:len(time_sim)-t];
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
                                
                force[n,t] = force[n,t] #*FL_temp*FV_temp;                
        
        #FP1 = F_pe_1_function(Lce/Lmax,Vce);
        #FP2 = F_pe_2_function(Lce);
        #if FP2 > 0:
        #    FP2 = 0;
        
        Force = np.sum(force[:,t])
        #+ FP1*F0 + FP2*F0                          
        
    
        #ForceSE_vec[t] = ForceSE;
        Force_vec[t] = Force;
        Lce_vec[t] = Lce;
        U_eff_vec[t] = U_eff;
        f_env_vec[t] = f_env[-1];
        Af_vec[t] = Af[-1];
        FF_vec[t] = FF[-1];
        S_vec_store[t] = S_vec[-1];
        
    end_time = time.time()
    print(end_time - start_time)
    
    output = {'Time':time_sim, 'Muscle Force':Force_vec, 'Input': Input,
              'Twitch Force': force, 'Spike Train':spike_train, 'Muscle Length': Lce_vec,
              'U_eff': U_eff_vec,'f_env':f_env_vec, 'Af':Af_vec,'FF':FF_vec,
              'S': S_vec_store}
              
    
    default_path = '/Users/akiranagamori/Documents/GitHub/python-code/';  
    save_path = '/Users/akiranagamori/Documents/GitHub/python-code/Data';          
    os.chdir(save_path)
    np.save('output_temp.npy',output)
    os.chdir(default_path)
    
    fig1 = plt.figure()
    ax1 = fig1.add_subplot(111)
    ax1.plot(time_sim,Force_vec)
    
    
#    fig2 = plt.figure()
#    ax2 = fig2.add_subplot(111)
#    ax2.plot(time_sim,Lce_vec)
      
    plt.show()
    plt.plot(time_sim,ForceSE_vec)
    return (Force_vec,ForceSE_vec);


(Force_vec,ForceSE_vec) = TwitchBasedMuscleModel()
#(Force,ForceSE) = MuscleModel()