#!/usr/bin/env python2
# -*- coding: utf-8 -*-
"""
Created on Mon Mar 12 13:59:01 2018

@author: akira
"""
from distutils.core import setup
from Cython.Build import cythonize
from distutils.extension import Extension

ext_modules=[ Extension("SDN_SingleTrial",
              ["SDN_SingleTrial.pyx"],
              libraries=["m"],
              extra_compile_args = ["-ffast-math"])]

setup(
    ext_modules = cythonize("SDN_SingleTrial.pyx")
) 