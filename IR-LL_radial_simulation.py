#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Tue May 23 09:46:26 2017

@author: omaier
"""

import numpy as np
import fibonacci
import matplotlib.pyplot as plt


NScan = 1200
Nproj = 1

T1 = np.array([600,3000])
M0 = 1
alpha = 5*np.pi/180
TR                = 10000-(6*Nproj*NScan+14.7)
tau = 6
td = 14.7



                      
                    
                
def IR_LL_model(i,M0,T1,alpha,N_LL,tau,TR,td):
  E_tau = np.exp(-tau/T1)
  E_r = np.exp(-TR/T1)
  E_d = np.exp(-td/T1)
  
  F = (1-E_tau)/(1-np.cos(alpha)*E_tau)
  Q = (-F*np.cos(alpha)*E_r*E_d*(1-(np.cos(alpha)*E_tau)**(N_LL-1))-2*E_d+E_r+1)/ \
      (1+np.cos(alpha)*E_r*E_d*(np.cos(alpha)*E_tau)**(N_LL-1))

  S_LL = np.sin(alpha)*M0*(F+(np.cos(alpha)*E_tau)**(i-1)*(Q-F))
  return S_LL



Ref = np.zeros((np.size(T1),NScan))
for i in range(NScan):
  Ref[:,i] = IR_LL_model(i,M0,T1,alpha,NScan,tau,TR,td)
  
  
series = fibonacci.run(14)
series = series[5:8]

S_comb = list()
S_mean = list()

for j in range(np.size(series)):
  Nproj = int(series[j])
  NScan = int(np.floor(1200/Nproj))

  tau = 6*Nproj
  td = 14.7+3*Nproj
  N_shots = NScan
  TR = 10000-(6*Nproj*NScan+14.7)
  tmp = np.zeros((np.size(T1),NScan))
  for i in range(NScan):
   tmp[:,i] = IR_LL_model(i,M0,T1,alpha,NScan,tau,TR,td)
  S_comb.append(tmp)

  tmp = np.reshape(Ref[:,:Nproj*NScan],[2,NScan,Nproj])
  S_mean.append(np.squeeze(np.mean(tmp,2)))

plt.close('all')

plt.figure(1)
plt.plot(Ref[0,:],label='Ref')     
for i in range(np.size(series)):
  plt.plot(np.arange(0,1200-int(series[i]*bool(np.mod(1200,series[i]))),int(series[i])),np.squeeze(S_mean[i][0,:]),label='Projections '+str(int(series[i])))    
plt.title('T1 = '+str(T1[0])+' With mean')
plt.legend()
  

plt.figure(2)
plt.plot(Ref[0,:],label='Ref')     
for i in range(np.size(series)):
  plt.plot(np.arange(0,1200-int(series[i]*bool(np.mod(1200,series[i]))),int(series[i])),np.squeeze(S_comb[i][0,:]),label='Projections '+str(int(series[i])))    
plt.title('T1 = '+str(T1[0])+' With apparent tau')
plt.legend()
  
               
               