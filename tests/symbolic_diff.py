#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Tue Jun  6 12:36:30 2017

@author: omaier
"""

from sympy import *



M0, M0_sc, T1,T1_sc,T2,T2_sc,TR,fa,fa_corr,n,TE = symbols('M0,M0_sc,T1,R10,T2,T2_sc,TR,fa,fa_corr,n,r')
#t, tau, mu_B,mu_G,A_G,Tc,alpha,Te,A_B,FP,FP_sc,Te_sc,alpha_sc = symbols('t, tau, mu_B,mu_G,A_G,Tc,alpha,Te,A_B,FP,FP_sc,Te_sc,alpha_sc')
init_printing(use_unicode=True)

#T2 = TE*T1+T1_sc
E1 = exp(-TR/(T1*T1_sc))
#E2 = exp(-TR/(T2*T2_sc))
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



S = M0*M0_sc*sin(fa*fa_corr)*(1-E1)/(1-E1*cos(fa*fa_corr))

#S = (M0*M0_sc*sin(fa*fa_corr)*(1-E1))/(1-(E1-E2)*cos(fa*fa_corr)-(E1*E2));

#S = M0*M0_sc*exp(-b*ADC*ADC_sc+1/6*b**2*(TE*(ADC*ADC_sc+f*f_sc))**2*kurt*kurt_sc)
#S = M0*M0_sc*(f*f_sc*exp(-b*(ADC*ADC_sc+ADC2*ADC2_sc))+(1-f*f_sc)*exp(-b*ADC*ADC_sc))

#S = bSSFP*(1-INV*exp(-n*TR/T1_star))

#S = M0*M0_sc*exp(-TE/(T2*T2_sc))
#
M0_grad = str((diff(S,M0)))
E1_grad = str((diff(S,T1)))
#E2_grad = str((diff(S,T2)))
#f_grad = str((diff(S,f)))


#
#
#t2 = t-tau
#t3 = 1.0/mu_B
#t4 = 1.0/mu_G
#t7 = mu_B*t2
#t5 = exp(-t7)
#t6 = 1.0/mu_B**2
#t8 = t3*t5
#t30 = mu_G*t2
#t9 = exp(-t30)
#t31 = t4*t9
#t10 = t8-t31
#t11 = A_G*t10
#t12 = Tc-t+tau
#t13 = t2*t3
#t14 = t6+t13
#t15 = exp(-(alpha*alpha_sc))
#t16 = t15-1.0
#t17 = 1.0/(Te*Te_sc)
#t18 = 1.0/(alpha*alpha_sc)
#t19 = t16*t17*t18
#t20 = mu_B+t19
#t21 = 1.0/t20
#t22 = mu_G+t19
#t23 = 1.0/t22
#t24 = t12*t20
#t25 = exp(t24)
#t26 = 1.0/t20**2
#t27 = mu_B*t12
#t28 = exp(t27)
##        t29 = 1.0/mu_B**3
##        t32 = 1.0/mu_G**2
#t42 = t12*t16*t17*t18
#t33 = exp(-t42)
#t44 = t12*t21
#t34 = t26-t44
##        t35 = t5*t6
##        t36 = t2*t3*t5
##        t37 = t35+t36
##        t38 = t29*2.0
##        t39 = t2*t6
##        t40 = t38+t39
##        t41 = A_B*t5*t40
#t43 = 1.0/t20**3
##        t45 = A_B*t2*t5*t14
#t46 = t3*t28
#t47 = mu_G*t12
#t48 = exp(t47)
#t49 = t21*t25
#t50 = t12*t22
#t51 = exp(t50)
##        t52 = t9*t32
##        t53 = t2*t4*t9
##        t54 = t52+t53
##        t55 = A_G*t54
#t56 = 1.0/t22**2
#t57 = A_B*t26
#t58 = t21-t23
#t59 = t23*t51
#t60 = 1.0/(Te*Te_sc)**2
#t61 = t16**2
#t62 = t49-t59
#t63 = A_G*t62
#t65 = A_G*t58
#t66 = A_B*t25*t34
#t64 = t57+t63-t65-t66
#t67 = 1.0/(alpha*alpha_sc)**2
#t68 = t16*t17*t67
#t69 = t15*t17*t18
#t70 = t68+t69
##        t71 = t5-t9
##        t72 = A_G*t71
##        t73 = t28-t48
##        t74 = A_B*t3*t5
##        t75 = t25-t51
##        t76 = A_G*t75
##        t77 = A_B*t21*t25
##        t78 = t76+t77-A_B*t20*t25*t34
##        t79 = A_B*mu_B*t28*(t6-t3*t12)
##        t80 = t17*t18*t33*t61*t64
#
#S1=(t11+A_B*t6-A_G*(t3-t4)-A_B*t5*t14)*FP*FP_sc
#
#
#S2=(t11-A_G*(t46-t4*t48)+A_B*t28*(t6-t3*t12)-t16*t33*(t57-A_G*t58+A_G*(t49-t23*t51)-\
#             A_B*t25*t34)-A_B*t5*t14)*FP*FP_sc
#
#
#FP_grad = str((diff(S2,FP)))
#Te_grad = str((diff(S2,Te)))
#alpha_grad = str((diff(S2,alpha)))
