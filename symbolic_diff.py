#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Tue Jun  6 12:36:30 2017

@author: omaier
"""

from sympy import *



M0, M0_sc, T1, T1_sc, sin,cos,TR,tau,td,N,n = symbols('M0,M0_sc,T1,T1_sc,sin,cos,TR,tau,td,N,n')
init_printing(use_unicode=True)

E1 = exp(-TR/(T1*T1_sc))
Etau = exp(-tau/(T1*T1_sc))
Etd = exp(-td/(T1*T1_sc))

F = (1-Etau)/(1-cos*Etau)
Q = (-F*cos*E1*Etd*(1-(cos*Etau)**(N-1))-2*Etd+E1+1)/(1+cos*E1*Etd*(cos*Etau)**(N-1))


S = M0*M0_sc*sin*(F+(cos*Etau)**(n-1)*(Q-F))

M0_grad = str((diff(S,M0)))
T1_grad = str((diff(S,T1)))
