#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Thu Jan 11 15:04:27 2018

@author: omaier
"""


import numpy as np
from tkinter import filedialog
from tkinter import Tk

import h5py  


import os
import dicom
import time

root = Tk()
root.withdraw()
root.update()
root_dir = filedialog.askdirectory()
root.destroy()

filenames = []

for root, dirs, files in os.walk(root_dir, topdown=False):
    for name in files:
        filenames.append((os.path.join(root, name)))
        
        
filenames.sort()
images=[]
for filename in filenames:
  images.append(dicom.read_file(filename).pixel_array)

images = np.array(images,np.float64)



################################################################################
### New .hdf5 save files #######################################################
################################################################################
outdir = time.strftime("%Y-%m-%d  %H-%M-%S_VFA_Reference")
if not os.path.exists('./output'):
    os.makedirs('./output')
os.makedirs("output/"+ outdir)

os.chdir("output/"+ outdir)

f = h5py.File("output_"+name,"w")
dset_T1=f.create_dataset("T1_ref",np.squeeze(images).shape,\
                         dtype=np.float64,\
                         data=np.squeeze(images))
dset_M0=f.create_dataset("M0_ref",np.squeeze(images).shape,\
                         dtype=np.float64,\
                         data=np.squeeze(images))
f.flush()
f.close()

os.chdir('..')
os.chdir('..')