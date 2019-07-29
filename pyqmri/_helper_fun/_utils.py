#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Wed Jan 30 11:10:07 2019

@author: omaier
"""
import numpy as np
import configparser
from pyqmri._transforms._pyopencl_nufft import PyOpenCLNUFFT

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


def NUFFT(par, trafo=1):
    NC = par["NC"]
    NSlice = par["NSlice"]
    par["NC"] = 1
    par["NSlice"] = 1
    FFT = PyOpenCLNUFFT(par["ctx"][0], par["queue"][0], par,
                        radial=trafo)
    par["NC"] = NC
    par["NSlice"] = NSlice
    return FFT


def gen_default_config():

    config = configparser.ConfigParser()

    config['3D_TGV'] = {}
    config['3D_TGV']["max_iters"] = '1000'
    config['3D_TGV']["start_iters"] = '10'
    config['3D_TGV']["max_gn_it"] = '7'
    config['3D_TGV']["lambd"] = '1e0'
    config['3D_TGV']["gamma"] = '1e-3'
    config['3D_TGV']["delta"] = '1e-4'
    config['3D_TGV']["omega"] = '0'
    config['3D_TGV']["display_iterations"] = '1'
    config['3D_TGV']["gamma_min"] = '3e-4'
    config['3D_TGV']["delta_max"] = '1e-1'
    config['3D_TGV']["omega_min"] = '0'
    config['3D_TGV']["tol"] = '1e-6'
    config['3D_TGV']["stag"] = '1e10'
    config['3D_TGV']["delta_inc"] = '10'
    config['3D_TGV']["gamma_dec"] = '0.7'
    config['3D_TGV']["omega_dec"] = '0.5'

    config['3D_TV'] = {}
    config['3D_TV']["max_iters"] = '1000'
    config['3D_TV']["start_iters"] = '10'
    config['3D_TV']["max_gn_it"] = '7'
    config['3D_TV']["lambd"] = '1e0'
    config['3D_TV']["gamma"] = '1e-3'
    config['3D_TV']["delta"] = '1e-4'
    config['3D_TV']["omega"] = '0'
    config['3D_TV']["display_iterations"] = '1'
    config['3D_TV']["gamma_min"] = '3e-4'
    config['3D_TV']["delta_max"] = '1e-1'
    config['3D_TV']["omega_min"] = '0'
    config['3D_TV']["tol"] = '1e-6'
    config['3D_TV']["stag"] = '1e10'
    config['3D_TV']["delta_inc"] = '10'
    config['3D_TV']["gamma_dec"] = '0.7'
    config['3D_TV']["omega_dec"] = '0.5'

    with open('default.ini', 'w') as configfile:
        config.write(configfile)


def read_config(conf_file, reg_type="DEFAULT"):
    config = configparser.ConfigParser()
    try:
        with open(conf_file + '.ini', 'r') as f:
            config.read_file(f)
    except BaseException:
        print("Config file not readable or not found. "
              "Falling back to default.")
        gen_default_config()
        with open('default.ini', 'r') as f:
            config.read_file(f)
    finally:
        params = {}
        for key in config[reg_type]:
            if key in {'max_gn_it', 'max_iters', 'start_iters'}:
                params[key] = int(config[reg_type][key])
            elif key == 'display_iterations':
                params[key] = config[reg_type].getboolean(key)
            else:
                params[key] = float(config[reg_type][key])
        return params
