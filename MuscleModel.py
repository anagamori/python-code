#!/usr/bin/env python2
# -*- coding: utf-8 -*-
"""
Created on Mon Jan 22 17:54:53 2018

@author: akiranagamori
"""

def MuscleModel():

    import time
    import numpy as np
    import matplotlib.pyplot as plt
    
    def yield_function(Y,V,step):
        c_y = 0.35;
        V_y = 0.1;
        T_y = 0.2;
        Y_dot = (1-c_y*(1-np.exp(-abs(V)/float(V_y)))-Y)/T_y;
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
            
    def f_slow_function(f_out,f_in,f_env,f_eff_dot,Af,Lce,step):
        T_f1 = 0.0343;
        T_f2 = 0.0227;
        T_f3 = 0.047;
        T_f4 = 0.0252;
        
        if f_eff_dot >= 0:
            Tf = T_f1*np.power(Lce,2)+T_f2*f_env;
        else:
            Tf = (T_f3 + T_f4*Af)/float(Lce);
            
        f_out_dot = (f_in - f_out)/Tf;
        f_out = f_out_dot*step + f_out;
        
        return (f_out,f_out_dot)
        
        
    def f_fast_function(f_out,f_in,f_env,f_eff_dot,Af,Lce,step):
        T_f1 = 0.0206;
        T_f2 = 0.0136;
        T_f3 = 0.0282;
        T_f4 = 0.0151;
        
        if f_eff_dot >= 0:
            Tf = T_f1*np.power(Lce,2)+T_f2*f_env;
        else:
            Tf = (T_f3 + T_f4*Af)/float(Lce);
            
        f_out_dot = (f_in - f_out)/Tf;
        f_out = f_out_dot*step + f_out;
        
        return (f_out,f_out_dot)
    
    def Af_slow_function(f_eff,L,Y):
        a_f = 0.56;
        n_f0 = 2.1;
        n_f1 = 5;
        n_f = n_f0 + n_f1*(1/float(L)-1);
        Af = 1 - np.exp(-np.power(Y*f_eff/float(a_f*n_f),n_f));
        return Af
    
    def Af_fast_function(f_eff,L,S):
        a_f = 0.56;
        n_f0 = 2.1;
        n_f1 = 3.3;
        n_f = n_f0 + n_f1*(1/float(L)-1);
        Af = 1 - np.exp(-np.power(S*f_eff/float(a_f*n_f),n_f));
        return Af
    
    def FL_function(L):
        beta = 2.3;
        omega = 1.12;
        rho = 1.62;
        
        FL = np.exp(-np.power(abs((np.power(L,beta) - 1)/float(omega)),rho));
        return FL
    
    def FL_fast_function(L):
        beta = 1.55;
        omega = 0.75;
        rho = 2.12;
        
        FL = np.exp(-np.power(abs((np.power(L,beta) - 1)/float(omega)),rho));
        return FL
    
    def FV_con_function(L,V):
        Vmax = -7.88;
        cv0 = 5.88;
        cv1 = 0;
        
        FVcon = (Vmax - V)/float(Vmax + (cv0 + cv1*L)*V);
        return FVcon
    
    def FV_con_fast_function(L,V):
        Vmax = -9.15;
        cv0 = -5.7;
        cv1 = 9.18;
        
        FVcon = (Vmax - V)/float(Vmax + (cv0 + cv1*L)*V);
        return FVcon
    
    def FV_ecc_function(L,V):
        av0 = -4.7;
        av1 = 8.41;
        av2 = -5.34;
        bv = 0.35;
        FVecc = (bv - (av0 + av1*L + av2*np.power(L,2))*V)/float(bv+V);
        
        return FVecc
    
    def FV_ecc_fast_function(L,V):
        av0 = -1.53;
        av1 = 0;
        av2 = 0;
        bv = 0.69;
        FVecc = (bv - (av0 + av1*L + av2*np.power(L,2))*V)/float(bv+V);
        
        return FVecc
    
    def F_pe_1_function(L,V):
        c1_pe1 = 23;
        k1_pe1 = 0.046;
        Lr1_pe1 = 1.17;
        eta = 0.01;
        
        Fpe1 = c1_pe1 * k1_pe1 * np.log(np.exp((L - Lr1_pe1)/float(k1_pe1))+1) + eta*V;
        
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
        
        Fse = cT_se * kT_se * np.log(np.exp((LT - LrT_se)/float(kT_se))+1);
        return Fse
    
    def InitialLength(L0,alpha,L_slack,Lce_initial,Lt_initial):
        cT = 27.8;
        kT = 0.0047;
        LrT = 0.964;
        c1 = 23;
        k1 = 0.046;
        Lr1 = 1.17;
        PassiveForce = c1 * k1 * np.log(np.exp((1 - Lr1)/float(k1))+1);
        Normalized_SE_Length = kT*np.log(np.exp(PassiveForce/float(cT)/float(kT))-1)+LrT;
        Lmt_temp_max = L0*np.cos(alpha)+L_slack + 1;
        L0_temp = L0;
        L0T_temp = L_slack*1.05;
        SE_Length =  L0T_temp * Normalized_SE_Length;
        FasclMax = (Lmt_temp_max - SE_Length)/float(L0_temp);
        Lmax = FasclMax/float(np.cos(alpha));
        Lmt_temp = Lce_initial * np.cos(alpha) + Lt_initial;
        InitialLength =  (Lmt_temp-(-L0T_temp*(kT/float(k1)*Lr1-LrT-kT*np.log(c1/float(cT)*k1/float(kT)))))/(100*(1+kT/float(k1)*L0T_temp/float(Lmax)*1/float(L0_temp))*np.cos(alpha));
        Lce_initial = InitialLength/float((L0_temp/100));
        Lse_initial = (Lmt_temp - InitialLength*np.cos(alpha)*100)/float(L0T_temp);
        
        return (Lce_initial,Lse_initial,Lmax)
    
        
    #import matplotlib.pyplot as plt
    # 
    L0 = 6.8;
    alpha = 9.6*np.pi/180;
    L_slack = 24.1;
    L0T = L_slack*1.05;
    Lce_initial = 6.8;
    Lt_initial = 24.1;
    Lmt = Lce_initial*np.cos(alpha)+Lt_initial;
    (Lce,Lse,Lmax) =  InitialLength(L0,alpha,L_slack,Lce_initial,Lt_initial);
    
    density = 1.06;    
    mass = 0.15;    
    PCSA = (mass*1000)/float(density*L0)
    sigma = 31.8;
    F0 = PCSA*sigma;
    Ur = 0.8;
    F_pcsa_slow = 0.5;
    U1_th = 0.01;
    U2_th = Ur*F_pcsa_slow;
    
    f_half = 8.5;
    fmin = 0.5*f_half;
    fmax = 2*f_half;
    
    f_half_fast = 34;
    fmin_fast = 0.5*f_half_fast;
    fmax_fast = 2*f_half_fast;
    
    start_time = time.time()
    Fs = 10000
    step = 1/float(Fs)
    h = step;
    amp = 0.2
    time_sim = np.arange(0,5,step)
    U = np.concatenate((np.zeros(1*Fs),amp/float(2)*np.arange(0,2,step),amp*np.ones(5*Fs-3*Fs)),axis = 0)
    # plt.plot(t,U)
    # plt.show()
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
    
    Af_slow = 0;
    Af_fast = 0;
    
    MuscleVelocity = 0;
    MuscleLength = Lce*L0/float(100);
    
    Force_vec = np.zeros(len(time_sim));
    ForceSE_vec = np.zeros(len(time_sim));
    Lce_vec = np.zeros(len(time_sim));
    U_eff_vec = np.zeros(len(time_sim));
    f_env_vec = np.zeros(len(time_sim));
    f_int_vec = np.zeros(len(time_sim));
    f_eff_vec = np.zeros(len(time_sim));
    Y_vec = np.zeros(len(time_sim));
    S_vec = np.zeros(len(time_sim));
    Af_slow_vec = np.zeros(len(time_sim));
    Af_fast_vec = np.zeros(len(time_sim));
    
    for t in xrange(len(time_sim)):
        if U[t] >= U_eff:
            T_U = 0.03;
        else:
            T_U = 0.15;
        
        U_eff_dot = (U[t]-U_eff)/float(T_U);
        U_eff = U_eff_dot*step + U_eff;
        
        if U_eff < U1_th:
            W1 = 0;
        elif U_eff < U2_th:
            W1 = (U_eff - U1_th)/float(U_eff - U1_th);
        else:
            W1 = (U_eff - U1_th)/(float(U_eff - U1_th) + float(U_eff - U2_th));
        if U_eff < U2_th:
            W2 = 0;
        else:
            W2 = (U_eff - U2_th)/(float(U_eff - U1_th) + float(U_eff - U2_th));
        
        if U_eff >=  U1_th:
            f_env_slow = (fmax-fmin)/float(1-U1_th)*(U_eff-U1_th)+fmin;
            f_env_slow = f_env_slow/f_half;
        else:
            f_env_slow = 0;  
        
        (f_int_slow,f_int_slow_dot) = f_slow_function(f_int_slow,f_env_slow,f_env_slow,f_eff_slow_dot,Af_slow,Lce,step);
        (f_eff_slow,f_eff_slow_dot) = f_slow_function(f_eff_slow,f_int_slow,f_env_slow,f_eff_slow_dot,Af_slow,Lce,step);
    
        if U_eff >= U2_th:
            f_env_fast = (fmax_fast-fmin_fast)/float(1-U2_th)*(U_eff-U2_th)+fmin_fast;
            f_env_fast = f_env_fast/f_half_fast;
        else:
            f_env_fast = 0;
            
        (f_int_fast,f_int_fast_dot) = f_fast_function(f_int_fast,f_env_fast,f_env_fast,f_eff_fast_dot,Af_fast,Lce,step);
        (f_eff_fast,f_eff_fast_dot) = f_fast_function(f_eff_fast,f_int_fast,f_env_fast,f_eff_fast_dot,Af_fast,Lce,step);
        
        Y = yield_function(Y,Vce,step)
        S = sag_function(S,f_eff_fast,step)    
        Af_slow = Af_slow_function(f_eff_slow,Lce,Y);
        Af_fast = Af_fast_function(f_eff_fast,Lce,S);      
        
        if Vce <= 0:
            FV1 = FV_con_function(Lce,Vce);
            FV2 = FV_con_fast_function(Lce,Vce);
        else:
            FV1 = FV_ecc_function(Lce,Vce);
            FV2 = FV_ecc_fast_function(Lce,Vce);
        
        FL1 = FL_function(Lce);
        FL2 = FL_function(Lce);
        FP1 = F_pe_1_function(Lce/float(Lmax),Vce);
        FP2 = F_pe_2_function(Lce);
        if FP2 > 0:
            FP2 = 0;
        
        Fce = U_eff*((W1*Af_slow*(FL1*FV1+FP2))+(W2*Af_fast*(FL2*FV2+FP2)));
        if Fce < 0:
            Fce = 0;
        elif Fce > 1:
            Fce = 1;
        
        Fce = Fce + FP1;
        
        Force = Fce*F0;
        
        ForceSE = F_se_function(Lse)*F0;
                     
        k_0 = h*MuscleVelocity;
        l_0 = h*((ForceSE*np.cos(alpha) - Force*np.power(np.cos(alpha),2))/float(mass) + (np.power(MuscleVelocity,2)*np.power(np.tan(alpha),2)/float(MuscleLength)));
        k_1 = h*(MuscleVelocity+l_0/2);
        l_1 = h*((ForceSE*np.cos(alpha) - Force*np.power(np.cos(alpha),2))/float(mass) + (np.power(MuscleVelocity+l_0/2,2)*np.power(np.tan(alpha),2)/float(MuscleLength+k_0/2)));
        k_2 = h*(MuscleVelocity+l_1/2);
        l_2 = h*((ForceSE*np.cos(alpha) - Force*np.power(np.cos(alpha),2))/float(mass) + (np.power(MuscleVelocity+l_1/2,2)*np.power(np.tan(alpha),2)/float(MuscleLength+k_1/2)));
        k_3 = h*(MuscleVelocity+l_2);
        l_3 = h*((ForceSE*np.cos(alpha) - Force*np.power(np.cos(alpha),2))/float(mass) + (np.power(MuscleVelocity+l_2,2)*np.power(np.tan(alpha),2)/float(MuscleLength+k_2)));
        MuscleLength = MuscleLength + 1/float(6)*(k_0+2*k_1+2*k_2+k_3);
        MuscleVelocity = MuscleVelocity + 1/float(6)*(l_0+2*l_1+2*l_2+l_3);
        
        Vce = MuscleVelocity/float(L0/100);
        Lce = MuscleLength/float(L0/100);
        Lse = (Lmt - Lce*L0*np.cos(alpha))/float(L0T);
        
        
        ForceSE_vec[t] = ForceSE;
        Force_vec[t] = Force;
        Lce_vec[t] = Lce;
        U_eff_vec[t] = U_eff;
        f_env_vec[t] = f_env_slow;
        f_int_vec[t] = f_int_slow;
        f_eff_vec[t] = f_eff_slow;
        Y_vec[t] = Y;
        S_vec[t] = S;
        Af_slow_vec[t] = Af_slow;
        Af_fast_vec[t] = Af_fast;
        
    end_time = time.time()
    print(end_time - start_time)
    
    fig1 = plt.figure()
    ax1 = fig1.add_subplot(111)
    ax1.plot(time_sim,ForceSE_vec)
    
    fig2 = plt.figure()
    ax2 = fig2.add_subplot(111)
    ax2.plot(time_sim,Lce_vec)
    
# =============================================================================
#     fig3 = plt.figure()
#     ax3_1 = fig3.add_subplot(211)
#     ax3_1.plot(time_sim,Af_slow_vec)
#     ax3_2 = fig3.add_subplot(212)
#     ax3_2.plot(time_sim,Af_fast_vec)
#     
#     fig4 = plt.figure()
#     ax4 = fig4.add_subplot(111)
#     ax4.plot(time_sim,U_eff_vec)
#     
#     fig5 = plt.figure()
#     ax5_1 = fig5.add_subplot(311)
#     ax5_1.plot(time_sim,f_env_vec)
#     ax5_2 = fig5.add_subplot(312)
#     ax5_2.plot(time_sim,f_int_vec)
#     ax5_3 = fig5.add_subplot(313)
#     ax5_3.plot(time_sim,f_eff_vec)
#     
#     fig6 = plt.figure()
#     ax6_1 = fig6.add_subplot(211)
#     ax6_1.plot(time_sim,Y_vec)
#     ax6_2 = fig6.add_subplot(212)
#     ax6_2.plot(time_sim,S_vec)
# =============================================================================
    
    #plt.show()
    #plt.plot(time_sim,force[1,])
    #plt.plot(time_sim,spike_train[1,])
    #plt.show()
    return (Force_vec,ForceSE_vec);



(Force,ForceSE) = MuscleModel()