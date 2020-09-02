#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Wed Jan 30 11:10:07 2019

@author: omaier
"""
import configparser
import os
import numpy as np
from pyqmri.transforms import PyOpenCLnuFFT
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


def NUFFT(par, trafo=True, SMS=False):
    NC = par["NC"]
    NScan = par["NScan"]
    par["NC"] = 1
    par["NScan"] = 1

    FFT = (PyOpenCLnuFFT.create(
        par["ctx"][0], par["queue"][0], par,
        radial=trafo, SMS=SMS))
    par["NC"] = NC
    par["NScan"] = NScan

    return FFT


def gen_default_config():

    config = configparser.ConfigParser()

    config['TGV'] = {}
    config['TGV']["max_iters"] = '1000'
    config['TGV']["start_iters"] = '10'
    config['TGV']["max_gn_it"] = '7'
    config['TGV']["lambd"] = '1e0'
    config['TGV']["gamma"] = '1e-3'
    config['TGV']["delta"] = '1e-4'
    config['TGV']["omega"] = '0'
    config['TGV']["display_iterations"] = '0'
    config['TGV']["gamma_min"] = '3e-4'
    config['TGV']["delta_max"] = '1e-1'
    config['TGV']["omega_min"] = '0'
    config['TGV']["tol"] = '1e-6'
    config['TGV']["stag"] = '1e10'
    config['TGV']["delta_inc"] = '10'
    config['TGV']["gamma_dec"] = '0.7'
    config['TGV']["omega_dec"] = '0.5'

    config['TV'] = {}
    config['TV']["max_iters"] = '1000'
    config['TV']["start_iters"] = '10'
    config['TV']["max_gn_it"] = '7'
    config['TV']["lambd"] = '1e0'
    config['TV']["gamma"] = '1e-3'
    config['TV']["delta"] = '1e-4'
    config['TV']["omega"] = '0'
    config['TV']["display_iterations"] = '0'
    config['TV']["gamma_min"] = '3e-4'
    config['TV']["delta_max"] = '1e-1'
    config['TV']["omega_min"] = '0'
    config['TV']["tol"] = '1e-6'
    config['TV']["stag"] = '1e10'
    config['TV']["delta_inc"] = '10'
    config['TV']["gamma_dec"] = '0.7'
    config['TV']["omega_dec"] = '0.5'

    with open('default.ini', 'w') as configfile:
        config.write(configfile)


def read_config(conf_file, reg_type="DEFAULT"):
    config = configparser.ConfigParser()

    if not conf_file.endswith('.ini'):
        conf_file += '.ini'
    try:
        with open(conf_file, 'r') as f:
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


def save_config(conf, path, reg_type="DEFAULT"):
    tmp_dict = {}
    tmp_dict[reg_type] = conf
    config = configparser.ConfigParser()
    config.read_dict(tmp_dict)
    with open(path+os.sep+'config.ini', 'w') as configfile:
        config.write(configfile)

def fibonacci(num):
    if num < 3:
        if num < 2:
            fib = 1
        else:
            np.zeros((2))
            fib[0] = 1
            fib[1] = 1
    else:
        fib = np.zeros(num)
        fib[0] = 1
        fib[1] = 1
        for ii in range(2, num):
            fib[ii] = fib[ii - 2] + fib[ii - 1]
    return fib
