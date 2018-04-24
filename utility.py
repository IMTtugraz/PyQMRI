#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Mon Apr 23 14:35:17 2018

@author: omaier
"""

import pyopencl as cl
from operator import attrgetter
from typing import List, Iterator


def get_default_device(use_gpu: bool = True) -> cl.Device:
    """
    Retrieves the GPU device with the most global memory if available, otherwise returns the CPU.
    :param use_gpu: Determines whether to obtain a GPU or CPU device
    """
    platforms = cl.get_platforms()
    gpu_devices = [plat.get_devices(cl.device_type.GPU) for plat in platforms]
    gpu_devices = [dev for devices in gpu_devices for dev in devices]  # Flatten to 1d if multiple GPU devices exists

    if gpu_devices and use_gpu:
        dev = max(gpu_devices, key=attrgetter('global_mem_size'))
        print('Using GPU: {}'.format(dev.name))
        print('On platform: {} ({})\n'.format(dev.platform.name, dev.platform.version.strip()))
        return dev
    else:
        cpu_devices = [plat.get_devices(cl.device_type.CPU) for plat in platforms]
        cpu_devices = [dev for devices in cpu_devices for dev in devices]
        if cpu_devices:
            dev = max(cpu_devices, key=attrgetter('global_mem_size'))
            print('Using CPU: {}'.format(dev.name))
            print('On platform: {} ({})\n'.format(dev.platform.name, dev.platform.version.strip()))
            return dev
        else:
            raise RuntimeError('No suitable OpenCL GPU/CPU devices found')


def get_devices_by_name(name: str, case_sensitive: bool = False) -> List[cl.Device]:
    """
    Searches through all devices looking for a partial match for 'name' among the available devices.
    :param name: The string to search for
    :param case_sensitive: If false, different case is ignored when searching
    :return: A list of all devices that is a partial match for the specified name
    """
    if not name:
        raise RuntimeError('Device name must be specified')

    platforms = cl.get_platforms()
    devices = [plat.get_devices(cl.device_type.ALL) for plat in platforms]
    devices = [dev for devices in devices for dev in devices]

    if case_sensitive:
        name_matches = [dev for dev in devices if name in dev.name]
    else:
        name_matches = [dev for dev in devices if name.lower() in dev.name.lower()]

    return name_matches


def range_bitwise_shift(low: int, high: int, n: int) -> Iterator[int]:
    """
    Generates an upwards or downwards range through successive bitshifts according to n.
    :param low: the lower part of the range (non-inclusive)
    :param high: the higher part of the range (non-inclusive)
    :param n: the number of times to perform the bitshift, can be negative
    :return: a generator of the bitshifted range
    """
    if not n:
        raise ValueError('n cannot be zero or None')

    if low > high:
        raise ValueError('low must have a value lower than high')

    if n > 0:
        i = low
        while i < high:
            yield i
            i <<= n
    else:
        i = high
        while i > low:
            yield i
            i >>= abs(n)
