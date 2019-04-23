#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Thu Nov 15 09:54:58 2018

@author: omaier

Copyright 2019 Oliver Maier

Licensed under the Apache License, Version 2.0 (the "License");
you may not use this file except in compliance with the License.
You may obtain a copy of the License at

    http://www.apache.org/licenses/LICENSE-2.0

Unless required by applicable law or agreed to in writing, software
distributed under the License is distributed on an "AS IS" BASIS,
WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
See the License for the specific language governing permissions and
limitations under the License.

"""
import numpy as np
from scipy.linalg import svd
from tkinter import filedialog
from tkinter import Tk
import h5py

DTYPE_real = np.float32

root = Tk()
root.withdraw()
root.update()
file = filedialog.askopenfilename()
root.destroy()

name = file.split('/')[-1]

file = h5py.File(file)


data = data = file['real_dat'][()] + 1j * file['imag_dat'][()]

print('performing coil compression ...')
[nscan, ncoils, rNz, rNy, rNx] = data.shape
svdcoil_fac = 0.1
data_svd = np.reshape(
    np.transpose(
        data.T, [
            0, 1, 2, 4, 3]), (rNy * rNx * rNz * nscan, ncoils))
[U, S, V] = svd(data_svd, full_matrices=False)
ncoils_svd = np.sum((S) / S[0] > svdcoil_fac)
print('using ' + str(ncoils_svd) + ' virtual channels ...')
data = np.transpose(np.reshape(
    data_svd@V.T[:, :ncoils_svd], (rNx, rNy, rNz, nscan, ncoils_svd)),
    [0, 1, 2, 4, 3]).T
del U, S, V, ncoils_svd, data_svd
print(' done \n')


del file['real_dat']
del file['imag_dat']

file.create_dataset(
    "real_dat",
    data.shape,
    dtype=DTYPE_real,
    data=np.real(data))
file.create_dataset(
    "imag_dat",
    data.shape,
    dtype=DTYPE_real,
    data=np.imag(data))

file.flush()
file.close()
