#!/bin/sh
set -exv
sudo apt-get install -y opencl-headers libclfft*
git clone https://github.com/geggo/gpyfft.git
pip install cython
cd gpyfft
pip install .
cd ..
