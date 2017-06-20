#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Tue Jun  6 12:36:30 2017

@author: omaier
"""

from sympy import *



M0, M0_sc, T1, T1_sc, sin,cos,TR = symbols('M0,M0_sc,T1,T1_sc,sin,cos,TR')
init_printing(use_unicode=True)

E1 = exp(-TR/(T1*T1_sc))
S = M0*M0_sc*exp(-TR/(T1*T1_sc))

M0_grad = str(diff(S,M0))
T1_grad = str((diff(S,T1)))
