#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Tue Jul 17 11:56:16 2018

@author: omaier
"""
import numpy as np
#
#	function y = kb(u,w,beta)
#
#	Computes the Kaiser-Bessel function used for gridding, namely
#
#	y = f(u,w,beta) = I0 [ beta*sqrt(1-(2u/w)^2) ]/w
#
#	where I0 is the zero-order modified Bessel function
#		of the first kind.
#
#	INPUT:
#		u = vector of k-space locations for calculation.
#		w = width parameter - see Jackson et al.
#		beta = beta parameter - see Jackson et al.
#
#	OUTPUT:
#		y = vector of Kaiser-Bessel values.
#

#	B. Hargreaves	Oct, 2003.


def kb(u,w,beta,G):
  if (np.size(w) > 1):
    	raise('w should be a single scalar value.');



  y = 0*u # Allocate space.
  uz = np.where(np.abs(u)< w/(2*G))			# Indices where u<w/2.

  if (np.size(uz) > 0):			# Calculate y at indices uz.
    	x = beta*np.sqrt(1-(2*G*u[uz]/w)**2);	# Argument - see Jackson '91.
    	y[uz] = G*np.i0(x)/w;

  ft_y = np.flip(y,0)
  ft_y = np.concatenate((ft_y,y),0)
  ft_y = np.pad(ft_y,int(G*u.size/w),'constant')
  ft_y = np.fft.ifftshift(np.fft.ifft(np.fft.fftshift(ft_y)))
  ft_y = ft_y[int(ft_y.size/2-G/4):int(ft_y.size/2+G/4)]
  ft_y = np.abs(ft_y/np.max(np.abs(ft_y)))
  return (y,ft_y)



