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
  beta = np.pi*np.sqrt( w**2/a**2*(a-0.5)**2-0.8 )	# From Beatty et al.



  u = np.linspace(0,klength-1,klength)/(klength-1) * kwidth/(2*G)     	# Kernel radii - grid samples.
#  u = (np.linspace(0,G-1,G)-G/2)/G

  (kern,kern_ft) = kb(u,kwidth, beta,G)
  kern=kern/kern[0]                 # Normalize.
  kbu = u
  return (kern,kern_ft,kbu)
