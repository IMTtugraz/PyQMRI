#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Tue Jun  6 12:36:30 2017

@author: omaier
"""

from sympy import *



M0, M0_sc, T1, T1_sc,T2,T2_sc,TR,fa,fa_corr,n,TE = symbols('M0,M0_sc,T1,T1_sc,T2,T2_sc,TR,fa,fa_corr,n,TE')
init_printing(use_unicode=True)


E1 = exp(-TR/(T1*T1_sc))
E2 = exp(-TR/(T2*T2_sc))
#T1_star = (1/(T1*T1_sc)*cos(fa/2)**2+1/(T2*T2_sc)*sin(fa/2)**2)**(-1)
#INV =  1 + (sin(fa/2)/sin(fa))*(((T1*T1_sc)/(T2*T2_sc)+1)-cos(fa)*((T1*T1_sc)/(T2*T2_sc)-1))
#bSSFP = M0*M0_sc*(sin(fa)*(1-E1))/(1-(E1-E2)*cos(fa)-(E1*E2))


#E1 = exp(-TR/(T1*T1_sc))#**(TR/1000)#exp(-TR/(T1*T1_sc))
#E2 = exp(-TR/(T2*T2_sc))#**(TR/1000)#exp(-TR/(T1*T1_sc))
#Etd = exp(-td/(T1*T1_sc))
#Etau = exp(-tau/(T1*T1_sc))
#Etau = Efit**(tau/100)
#Etd = Etau**(td/tau)
#E1 = Etau**(TR/tau)

#F = (1-Etau)/(1-cos(fa*fa_corr)*Etau)
#
#Q = (-F*cos(fa*fa_corr)*E1*Etd*(1-(cos(fa*fa_corr)*Etau)**(N-1))-2*Etd+E1*Etd+1)/(1+cos(fa*fa_corr)*E1*Etd*(cos(fa*fa_corr)*Etau)**(N-1))
#
#
#S = (M0*M0_sc*sin(fa*fa_corr)*(F+(cos(fa*fa_corr)*Etau)**(n-1)*(Q-F)))



#S = M0*M0_sc*sin(fa*fa_corr)*(1-E1)/(1-E1*cos(fa*fa_corr))

S = (M0*M0_sc*sin(fa*fa_corr)*(1-E1))/(1-(E1-E2)*cos(fa*fa_corr)-(E1*E2));

#S = M0*M0_sc*exp(-b*ADC*ADC_sc+1/6*b**2*(TE*(ADC*ADC_sc+f*f_sc))**2*kurt*kurt_sc)
#S = M0*M0_sc*(f*f_sc*exp(-b*(ADC*ADC_sc+ADC2*ADC2_sc))+(1-f*f_sc)*exp(-b*ADC*ADC_sc))

#S = bSSFP*(1-INV*exp(-n*TR/T1_star))

#S = M0*M0_sc*exp(-TE/(T2*T2_sc))

M0_grad = str((diff(S,M0)))
E1_grad = str((diff(S,T1)))
E2_grad = str((diff(S,T2)))
#f_grad = str((diff(S,f)))
