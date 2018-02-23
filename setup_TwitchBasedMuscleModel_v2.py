#!/usr/bin/env python2
# -*- coding: utf-8 -*-
"""
Created on Tue Feb 20 16:33:17 2018

@author: akiranagamori
"""

from distutils.core import setup
from Cython.Build import cythonize
from distutils.extension import Extension


ext_modules=[ Extension("TwitchBasedMuscleModel_v2",
              ["TwitchBasedMuscleModel_v2.pyx"],
              libraries=["m"],
              extra_compile_args = ["-ffast-math"])]

setup(
    ext_modules = cythonize("TwitchBasedMuscleModel_v2.pyx")
) 