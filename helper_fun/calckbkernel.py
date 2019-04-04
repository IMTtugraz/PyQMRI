#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Tue Jul 17 12:05:45 2018

@author: omaier
"""
import numpy as np
from helper_fun.kb import kb
#
#	function [kern,kbu] = calckbkernel(kwidth,overgridfactor,klength)
#
#	Function calculates the appropriate Kaiser-Bessel kernel
#	for gridding, using the approach of Jackson et al.
#
#	INPUT:
#		kwidth = kernel width in grid samples.
#		overgridfactor = over-gridding factor.
#		klength = kernel look-up-table length.
#
#	OUTPUT:
#		kern = kernel values for klength values
#			of u, uniformly spaced from 0 to kwidth/2.
#		kbu = u values.

#	B. Hargreaves



def calckbkernel(kwidth,overgridfactor,G,klength=32):


  if (klength < 2):
    	klength = 2
    	print('Warning:  klength must be 2 or more - using 2.')

  #beta = pi*kwidth*(overgridfactor-0.5);		# From Jackson et al.
  a = overgridfactor
  w = kwidth
  beta = np.pi*np.sqrt( w**2/a**2*(a-0.5)**2-0.8)	# From Beatty et al.


  u = np.linspace(0,np.floor(klength*w/2),int(np.ceil(klength*w/2)))/(np.floor(klength*w/2))*w/2/G     	# Kernel radii - grid samples.

  kern = kb(u,kwidth, beta, G)
  kern=kern/kern[u==0]                # Normalize.

  ft_y = np.flip(kern)
  ft_y = np.concatenate((ft_y[:-1],kern))
  ft_y = np.pad(ft_y,int(int(G*klength-ft_y.size)/2),'constant')
  ft_y = np.abs(np.fft.fftshift(np.fft.ifft(np.fft.ifftshift(ft_y)))*ft_y.size)
  ft_y = ft_y[int(ft_y.size/2-np.floor(G/(2*a))):int(ft_y.size/2+np.floor(G/(2*a)))]

  x = np.linspace(-int(G/(2*a)),int(G/(2*a))-1,int(G/(a)))
  h = np.sinc(x/(G*klength))**2
#  x = np.linspace(-int(G/(2*a)),int(G/(2*a))-1,int(G/(a)))
#  h = np.sqrt(2/3+1/3*np.cos(2*np.pi*x/(G*klength)))

  kern_ft = (ft_y*h)
  kern_ft = (kern_ft/np.max((kern_ft)))

  return (kern,kern_ft,u)
