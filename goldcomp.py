#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Thu Jan 18 12:05:59 2018

@author: omaier
"""
import numpy as np
from scipy.spatial import Voronoi


def cmp(k,cmp_type=None):
  if len(np.shape(k))==2:
    nspokes,N = np.shape(k)
  elif len(np.shape(k)) ==3:
    NScan,nspokes,N = np.shape(k)
  else:
    return -5

  if cmp_type == 'voronoi':
    fac=N*np.pi/2/nspokes   ### 4???
    w = DoCalcDCF(np.real(k), np.imag(k))
    w = fac*w.T
    w = reshape(w, [N, nspokes])
  else:
    w = np.abs(np.linspace(-N/2,N/2,N))  ### -N/2 N/2
    w = w*(np.pi/4)/nspokes   ### no scaling seems to work better??
    w = np.repeat(w,nspokes,0)
    w = np.reshape(w,(N,nspokes)).T

  return w


# Copyright (c) 2002, 2017 Jens Keiner, Stefan Kunis, Daniel Potts
#
# This program is free software; you can redistribute it and/or modify it under
# the terms of the GNU General Public License as published by the Free Software
# Foundation; either version 2 of the License, or (at your option) any later
# version.
#
# This program is distributed in the hope that it will be useful, but WITHOUT
# ANY WARRANTY; without even the implied warranty of MERCHANTABILITY or FITNESS
# FOR A PARTICULAR PURPOSE.  See the GNU General Public License for more
# details.
#
# You should have received a copy of the GNU General Public License along with
# this program; if not, write to the Free Software Foundation, Inc., 51
# Franklin Street, Fifth Floor, Boston, MA 02110-1301, USA.
#def precompute_weights( kx, ky ):
#
#  #input=load(file);
#
#  #kx = input(1:M,1);
#  #ky = input(1:M,2);
#
#  kxy= np.array(kx, ky)
#
#  # compute the voronoi regions
#  [V,C] = voronoin(kxy,{'QJ'})
#
#  # the surface of the knots is written to area
#  area = [];
#
#  # sum of all surfaces
#  sum_area = 0;
#
#  # the maximum distance two nearest neighbour have
#  # to get the surface we store max_distance^2
#  max_distance=0;
#
#  # compute the surface of the knots
#  for j in range(kxy.size):
#    x = V(C{j},1);
#    y = V(C{j},2);
#    lxy = length(x);
#    if(lxy==0) # a knot exists more than one time
#      A=0;
#    else
#      A = abs(sum(0.5*(x([2:lxy 1])-x(:)).* ...
#          (y([2:lxy 1]) + y(:))));
#    end
#    area = [area A];
#    min_distance = min((2*(x-kxy(j,1))).^2+(2*(y-kxy(j,2))).^2);
#    max_distance = max([max_distance min_distance]);
#  end
#
#  # if the surface of a knot is bigger than max_distance^2
#  # or isnan or isinf, then take max_distance^2
#  for j=1:length(area),
#    if(isnan(area(j)) | isinf(area(j))| area(j)>max_distance),
#      area(j)=max_distance;
#    end
#    sum_area = sum_area + area(j);
#  end
#
#  # norm the weights
#  area = area / sum_area;
#
#  return area

def DoCalcDCF(Kx, Ky):
    # caluclate density compensation factor using Voronoi diagram

    # remove duplicated K space points (likely [0,0]) before Voronoi
    K = Kx.flatten() + 1j*Ky.flatten()
    [K1,m1,n1]=np.unique(K,return_index=True,return_counts=True)
    K = K(np.sort(m1))

    # calculate Voronoi diagram
    [K2,m2,n2]=np.unique(K,return_index=True,return_counts=True)
    Kx = np.real(K2)
    Ky = np.imag(K2)
    Area = voronoiarea(Kx,Ky)

    # use area as density estimate
    DCF = Area(n1)

    # take equal fractional area for repeated locations (likely [0,0])
    # n   = n1;
    # while ~isempty(n)
    #     rep = length(find(n==n(1)));
    #     if rep > 1
    #         DCF (n1==n(1)) = DCF(n1==n(1))./rep;
    #     end
    #     n(n==n(1))=[];
    # end

    # normalize DCF
    DCF = DCF / np.max(DCF);

    # figure; voronoi(Kx,Ky);

    return DCF


def voronoiarea(Kx,Ky):
    # caculate area for each K space point as density estimate

    Kxy = [Kx,Ky]
    # returns vertices and cells of voronoi diagram
#    [V,C]  = Voronoin(Kxy)
    vor = Voronoin(Kxy)

    # compute area of each ploygon
    Area = np.zeros((1,(Kx).size))
    for j in range(Kx.size):
#        x = V(C{j},1)
#        y = V(C{j},2)
        # remove vertices outside K space limit including infinity vertices from voronoin
        x1 = x
        y1 = y
#        ind=find((x1.^2 + y1.^2)>0.25)
        x[ind]=[]
        y[ind]=[]
        # calculate area
        lxy = length(x)
        if lxy > 2 :
          test
#            ind=[2:lxy 1]
#            A = abs(sum(0.5*(x(ind)-x(:)).*(y(ind)+y(:))))
        else:
            A = 0
        Area[j] = A
    return Area
