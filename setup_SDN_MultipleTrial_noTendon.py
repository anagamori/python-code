#!/usr/bin/env python2
# -*- coding: utf-8 -*-
"""
Created on Tue Mar 13 19:01:34 2018

@author: akira
"""

from distutils.core import setup
from Cython.Build import cythonize
from distutils.extension import Extension

ext_modules=[ Extension("SDN_MultipleTrial_noTendon",
              ["SDN_MultipleTrial_noTendon.pyx"],
              libraries=["m"],
              extra_compile_args = ["-ffast-math"])]

setup(
    ext_modules = cythonize("SDN_MultipleTrial_noTendon.pyx")
) 