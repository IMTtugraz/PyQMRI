#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Tue Jun  6 12:36:30 2017

@author: omaier
"""

from sympy import *



M0, M0_sc, T1, fa,fa_corr,TR,tau,td,N,n,beta,Etau = symbols('M0,M0_sc,T1,fa,fa_corr,TR,tau,td,N,n,beta,Etau')
init_printing(use_unicode=True)

#E1 = exp(-TR/(T1*T1_sc))#**(TR/1000)#exp(-TR/(T1*T1_sc))
#Etau = exp(-tau/(T1*T1_sc))
#Etd = exp(-td/(T1*T1_sc))
#Etau = Efit**(tau/100)
Etd = Etau**(td/tau)
E1 = Etau**(TR/tau)

F = (1-Etau)/(1-cos(fa*fa_corr)*Etau)

Q = (-F*cos(fa*fa_corr)*E1*Etd*(1-(cos(fa*fa_corr)*Etau)**(N-1))-2*Etd+E1*Etd+1)/(1+cos(fa*fa_corr)*E1*Etd*(cos(fa*fa_corr)*Etau)**(N-1))


S = (M0*M0_sc*sin(fa*fa_corr)*(F+(cos(fa*fa_corr)*Etau)**(n-1)*(Q-F)))

#S = M0*M0_sc*sin(fa*fa_corr)*(1-E1)/(1-E1*cos(fa*fa_corr))

M0_grad = str((diff(S,M0)))
Etau_grad = str((diff(S,Etau)))
FA_grad = str((diff(S,fa_corr)))