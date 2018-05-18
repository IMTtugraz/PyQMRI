#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Tue Jun  6 12:36:30 2017

@author: omaier
"""

from sympy import *

init_printing(use_unicode=True)


M0, M0_sc, T1, T1_sc,fa,fa_corr,TR,tau,td,N,n,beta,scale = symbols('M0,M0_sc,T1,T1_sc,fa,fa_corr,TR,tau,td,N,n,beta,scale')

#E1 = exp(-TR/(T1*T1_sc))#**(TR/1000)#exp(-TR/(T1*T1_sc))
#Etau = exp(-tau/(T1*T1_sc))
Efit = T1*T1_sc
#Etd = Etau**(td/tau)#exp(-td/(T1*T1_sc))
Etau = Efit**(tau/scale)
Etd = Efit**(td/scale)
E1 = Efit**(TR/scale)

F = (1-Etau)/(1-cos(fa*fa_corr)*Etau)
#
Q = (-F*cos(fa*fa_corr)*E1*Etd*(1-(cos(fa*fa_corr)*Etau)**(N-1))-2*Etd+E1*Etd+1)/(1+cos(fa*fa_corr)*E1*Etd*(cos(fa*fa_corr)*Etau)**(N-1))
#
#
S = (M0*M0_sc*sin(fa*fa_corr)*(F+(cos(fa*fa_corr)*Etau)**(n-1)*(Q-F)))

#S = M0*M0_sc*sin(fa*fa_corr)*(1-E1)/(1-E1*cos(fa*fa_corr))

M0_grad = str((diff(S,M0)))
Etau_grad = str((diff(S,T1)))
FA_grad = str((diff(S,fa_corr)))
