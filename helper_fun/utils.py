#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Wed Jan 30 11:10:07 2019

@author: omaier
"""
import numpy as np
import configparser
from Transforms.gridroutines import gridding

DTYPE = np.complex64
DTYPE_real = np.float32

def prime_factors(n):
    i = 2
    factors = []
    while i * i <= n:
        if n % i:
            i += 1
        else:
            n //= i
            factors.append(i)
    if n > 1:
        factors.append(n)
    return factors

def NUFFT(par,trafo=1,SMS=0):
  NC = par["NC"]
  NSlice = par["NSlice"]
  par["NC"] = 1
  par["NSlice"] = 1
  FFT = gridding(par["ctx"][0],par["queue"],par,radial=trafo,SMS=SMS)
  par["NC"] = NC
  par["NSlice"] = NSlice
  return FFT

def gen_default_config():

  config = configparser.ConfigParser()

  config['DEFAULT'] = {}
  config['DEFAULT']["max_iters"] = '300'
  config['DEFAULT']["start_iters"] = '100'
  config['DEFAULT']["max_gn_it"] = '13'
  config['DEFAULT']["lambd"] = '1e2'
  config['DEFAULT']["gamma"] = '1e0'
  config['DEFAULT']["delta"] = '1e-1'
  config['DEFAULT']["display_iterations"] = 'True'
  config['DEFAULT']["gamma_min"] = '0.18'
  config['DEFAULT']["delta_max"] = '1e2'
  config['DEFAULT']["tol"] = '5e-3'
  config['DEFAULT']["stag"] = '1'
  config['DEFAULT']["delta_inc"] = '2'
  config['DEFAULT']["gamma_dec"] = '0.7'

  config['3D_TGV'] = {}
  config['3D_TGV']["max_iters"] = '1000'
  config['3D_TGV']["start_iters"] = '100'
  config['3D_TGV']["max_gn_it"] = '50'
  config['3D_TGV']["lambd"] = '1e6'
  config['3D_TGV']["gamma"] = '1e-1'
  config['3D_TGV']["delta"] = '1e-1'
  config['3D_TGV']["omega"] = '0'
  config['3D_TGV']["display_iterations"] = '1'
  config['3D_TGV']["gamma_min"] = '1e-4'
  config['3D_TGV']["delta_max"] = '1e2'
  config['3D_TGV']["omega_min"] = '0'
  config['3D_TGV']["tol"] = '5e-3'
  config['3D_TGV']["stag"] = '1e10'
  config['3D_TGV']["delta_inc"] = '5'
  config['3D_TGV']["gamma_dec"] = '0.7'
  config['3D_TGV']["omega_dec"] = '0.7'

  config['3D_TV'] = {}
  config['3D_TV']["max_iters"] = '300'
  config['3D_TV']["start_iters"] = '100'
  config['3D_TV']["max_gn_it"] = '13'
  config['3D_TV']["lambd"] = '1e2'
  config['3D_TV']["gamma"] = '2e-3'
  config['3D_TV']["delta"] = '1e-1'
  config['3D_TV']["display_iterations"] = 'True'
  config['3D_TV']["gamma_min"] = '0.8e-3'
  config['3D_TV']["delta_max"] = '1e2'
  config['3D_TV']["tol"] = '5e-3'
  config['3D_TV']["stag"] = '1'
  config['3D_TV']["delta_inc"] = '2'
  config['3D_TV']["gamma_dec"] = '0.7'

  with open('default.ini', 'w') as configfile:
    config.write(configfile)

def read_config(conf_file,reg_type="DEFAULT"):
  config = configparser.ConfigParser()
  try:
    with open(conf_file+'.ini','r') as f:
      config.read_file(f)
#    config.read(conf_file+".ini")
  except:
    print("Config file not readable or not found. Falling back to default.")
    gen_default_config()
    with open(conf_file+'.ini','r') as f:
      config.read_file(f)
  finally:
    params = {}
    for key in config[reg_type]:
      if key in {'max_gn_it','max_iters','start_iters'}:
        params[key] = int(config[reg_type][key])
      elif key == 'display_iterations':
        params[key] = config[reg_type].getboolean(key)
      else:
        params[key] = float(config[reg_type][key])
    return params