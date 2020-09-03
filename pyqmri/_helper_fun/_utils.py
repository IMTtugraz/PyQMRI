#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""Utility functions for PyQMRI fitting."""
import configparser
import os
from pyqmri.transforms import PyOpenCLnuFFT


def prime_factors(n):
    """Prime factorication.

    Parameters
    ----------
      n : int
        Value which should be factorized into primes.
    """
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
    """NUFFT for image guess.

    Parameters
    ----------
      par : dict
        Parameter struct to setup the NUFFT
      trafo : bool, True
        Radial (True) or Cartesian (False) FFT.
      SMS : bool, False
        SMS (True) or normal FFT (False, default).
    """
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
    """Generate default config file."""
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


def read_config(conf_file, reg_type="TGV"):
    """Config file reader.

    Parameters
    ----------
      conf_file : str
        Path to config file
      reg_type : str, TGV
        Select witch regularization parameters from the file should be used.
    """
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


def save_config(conf, path, reg_type="TGV"):
    """Config file writer.

    Save the used config alongside the results.

    Parameters
    ----------
      conf : str
        Path to config file
      path : str
        Output path
      reg_type : str, TGV
        Select witch regularization parameters from the file should be used.
    """
    tmp_dict = {}
    tmp_dict[reg_type] = conf
    config = configparser.ConfigParser()
    config.read_dict(tmp_dict)
    with open(path+os.sep+'config.ini', 'w') as configfile:
        config.write(configfile)
