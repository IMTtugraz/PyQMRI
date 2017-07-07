#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Tue May 23 09:46:26 2017

@author: omaier
"""

import numpy as np
import fibonacci
import matplotlib.pyplot as plt
import exp_mean
import scipy


NScan = 1200
N_LL = NScan
Nproj = 1

T1 = np.array([600,5000])
M0 = 1
alpha = 5*np.pi/180
TR                = 10000-(6*Nproj*NScan+14.7)
tau = 6
td = 14.7



                      
                    
                
def IR_LL_model(x,alpha,N_LL,tau,TR,td):
  
  E_tau = np.exp(-tau/x[1]/5000)
  E_r = np.exp(-TR/x[1]/5000)
  E_d = np.exp(-td/x[1]/5000)
  
  F = (1-E_tau)/(1-np.cos(alpha)*E_tau)
  Q = (-F*np.cos(alpha)*E_r*E_d*(1-(np.cos(alpha)*E_tau)**(N_LL-1))-2*E_d+E_r+1)/ \
      (1+np.cos(alpha)*E_r*E_d*(np.cos(alpha)*E_tau)**(N_LL-1))
  S_LL = np.zeros((np.size(x[1]),N_LL))
  for i in range(N_LL):
    S_LL[:,i] = np.sin(alpha)*x[0]*(F+(np.cos(alpha)*E_tau)**(i-1)*(Q-F))
  return S_LL

def cost_fun_mean(x,alpha,N_LL,tau,TR,td,NScan,Nproj,data):
  full = IR_LL_model(x,alpha,N_LL,tau,TR,td)
  full = np.reshape(full[:Nproj*NScan],[NScan,Nproj])
  return 1/2*np.linalg.norm(np.mean(full,-1)-data)**2

def cost_fun_exp(x,alpha,N_LL,tau,TR,td,NScan,Nproj,data):
  full = IR_LL_model(x,alpha,N_LL,tau,TR,td)
  full = np.reshape(full[:Nproj*NScan],[NScan,Nproj])
  return 1/2*np.linalg.norm(exp_mean.exp_mean(full,6,2500)-data)**2


Ref = np.zeros((np.size(T1),NScan))
for i in range(2):
 Ref[i,:] = IR_LL_model([M0,T1[i]],alpha,NScan,tau,TR,td)

#Ref = np.zeros((np.size(T1),NScan))
#for i in range(NScan):
#  Ref[:,i] = IR_LL_model(i,M0,T1,alpha,NScan,tau,TR,td)
  
  
series = fibonacci.run(14)
series = series[5:12]

S_comb = list()
S_mean_exp = list()
S_mean = list()
for j in range(np.size(series)):
  Nproj = int(series[j])
  NScan = int(np.floor(1200/Nproj))

  tau = 6*Nproj
  td = 14.7+3*Nproj
  N_shots = NScan
  TR = 10000-(6*Nproj*NScan+14.7)
  tmp = np.zeros((np.size(T1),NScan))
#  for i in range(NScan):
  for i in range(2):
   tmp[i,:] = IR_LL_model([M0,T1[i]],alpha,NScan,tau,TR,td)
  S_comb.append(tmp)

  tmp = np.reshape(Ref[:,:Nproj*NScan],[2,NScan,Nproj])
  S_mean.append(np.squeeze(np.mean(tmp,2)))
  S_mean_exp.append(np.squeeze(exp_mean.exp_mean(tmp,6,T1)))

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
  
      
plt.figure(3)
plt.plot(Ref[0,:],label='Ref')     
for i in range(np.size(series)):
  plt.plot(np.arange(0,1200-int(series[i]*bool(np.mod(1200,series[i]))),int(series[i])),np.squeeze(S_mean_exp[i][0,:]),label='Projections '+str(int(series[i])))    
plt.title('T1 = '+str(T1[0])+' With expt weights')
plt.legend()         
               
plt.figure(4)
for i in range(np.size(series)):
  plt.plot(np.arange(0,1200-int(series[i]*bool(np.mod(1200,series[i]))),int(series[i])),np.squeeze(S_mean[i][0,:]-S_mean_exp[i][0,:]),label='Projections '+str(int(series[i])))    
plt.title('T1 = '+str(T1[0])+' Diff_mean_exp')
plt.legend()       


test_exp = np.zeros((100,2))
test_mean = np.zeros((100,2))
for i in range(100):                
  test_exp[i,:] = scipy.optimize.fmin_cg(cost_fun_exp,[0.5,0.5],args=(alpha,N_LL,tau,TR,td,
      NScan,Nproj,S_mean_exp[0][0,:]+0.001*np.random.random_sample(np.shape(S_mean_exp[0][1,:]))),disp=False)
  test_mean[i,:] = scipy.optimize.fmin_cg(cost_fun_mean,[0,0.5],args=(alpha,N_LL,tau,TR,td,
      NScan,Nproj,S_mean[0][0,:]+0.001*np.random.random_sample(np.shape(S_mean[0][1,:]))),disp=False)
  
  
  