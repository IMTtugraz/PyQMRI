#!/bin/sh
set -ex
sudo apt-get update -qq
sudo apt-get install opencl-headers libclfft*
#sudo apt-get install poc
git clone https://github.com/geggo/gpyfft.git
cd gpyfft
pip install .
