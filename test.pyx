#!/usr/bin/env python2
# -*- coding: utf-8 -*-
"""
Created on Wed Feb 21 13:08:50 2018

@author: akiranagamori
"""

import time
import numpy as np

def sum_function(arr,n):
    summation = 0
    for i in xrange(n):
        summation += arr[i];
    return summation

Fs = 10000;
array1 = np.ones(5*Fs);

start_time = time.time()
array = sum_function(array1,len(array1))
end_time = time.time()
print(end_time - start_time)
print(array)

start_time = time.time()
array = np.sum(array1)
end_time = time.time()
print(end_time - start_time)
print(array)