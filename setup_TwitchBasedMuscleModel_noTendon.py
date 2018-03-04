#!/usr/bin/env python2
# -*- coding: utf-8 -*-
"""
Created on Sat Mar  3 21:55:49 2018

@author: akiranagamori
"""

from distutils.core import setup
from Cython.Build import cythonize
from distutils.extension import Extension


ext_modules=[ Extension("TwitchBasedMuscleModel_noTendon",
              ["TwitchBasedMuscleModel_noTendon.pyx"],
              libraries=["m"],
              extra_compile_args = ["-ffast-math"])]

setup(
    ext_modules = cythonize("TwitchBasedMuscleModel_noTendon.pyx")
) 