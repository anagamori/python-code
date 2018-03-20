#!/usr/bin/env python2
# -*- coding: utf-8 -*-
"""
Created on Tue Mar 20 09:22:22 2018

@author: akiranagamori
"""
#cython: boundscheck=False, wraparound=False, nonecheck=False

import numpy as np
from cython.parallel import prange
from libc.math cimport exp as c_exp

def c_array_f(double[:] X):

    cdef int N = X.shape[0]
    cdef double[:] Y = np.zeros(N)
    cdef int i

    for i in prange(N, nogil = True):
        if X[i] > 0.5:
            Y[i] = c_exp(X[i])
        else:
            Y[i] = 0

    return Y
