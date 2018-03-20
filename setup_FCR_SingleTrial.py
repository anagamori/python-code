#!/usr/bin/env python2
# -*- coding: utf-8 -*-
"""
Created on Mon Mar 12 09:50:59 2018

@author: akira
"""

from distutils.core import setup
from Cython.Build import cythonize
from distutils.extension import Extension

ext_modules=[ Extension("FCR_SingleTrial",
              ["FCR_SingleTrial.pyx"],
              libraries=["m"],
              extra_compile_args = ["-ffast-math"])]

setup(
    ext_modules = cythonize("FCR_SingleTrial.pyx")
) 