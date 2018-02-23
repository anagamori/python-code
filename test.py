#!/usr/bin/env python2
# -*- coding: utf-8 -*-
"""
Created on Wed Feb 21 13:04:26 2018

@author: akiranagamori
"""

import time
import numpy as np

Fs = 10000;
array1 = np.ones(5*Fs);

start_time = time.time()
array = np.sum(array1)
end_time = time.time()
print(end_time - start_time)