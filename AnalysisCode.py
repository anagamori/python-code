#!/usr/bin/env python2
# -*- coding: utf-8 -*-
"""
Created on Fri Feb 23 10:24:55 2018

@author: akiranagamori
"""

import numpy as np
import matplotlib.pyplot as plt
import os

default_path = '/Users/akiranagamori/Documents/GitHub/python-code/';  
save_path = '/Users/akiranagamori/Documents/GitHub/python-code/Data';  
        
os.chdir(save_path)
output = np.load('output.npy').item()
os.chdir(default_path)

Fs = 10000;
meanForce = np.mean(output['Tendon Force'][4*Fs:])
CoVForce = np.std(output['Tendon Force'][4*Fs:])/meanForce

plt.plot(output['C'])