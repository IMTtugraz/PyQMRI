#!/bin/sh
set -ex
#sudo apt-get install poc
git clone https://github.com/geggo/gpyfft.git
cd gpyfft
pip install .
