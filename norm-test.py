#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Tue Apr  3 09:49:10 2018

@author: omaier
"""


import gradients_divergences as gd

data = opt_t.data
x = opt.result[1,...]
x_new = x
xk = opt.result[1,...]
dz = 1
#v_new = opt.v
#v = opt.v

res = data - opt_t.FT(model.execute_forward_3D(x_new)) + opt_t.operator_forward_3D(x)
Ax = opt_t.operator_forward_3D(x_new)
gradx = gd.fgrad_3(x_new,1,1,dz)
#symgrad_v = gd.sym_bgrad_3(v_new,1,1,dz)

alpha = opt.irgn_par.gamma
beta = opt.irgn_par.gamma*2

delta = opt.irgn_par.delta

data = opt.irgn_par.lambd/2*np.linalg.norm((Ax-res).flatten())**2
TGV = alpha*np.sum(np.abs((gradx[:opt_t.unknowns_TGV]))) #+  beta*np.sum(np.abs(symgrad_v)) 
step = 1/(2*delta)*np.linalg.norm((x_new-xk).flatten())**2

data/TGV