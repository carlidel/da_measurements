#!/bin/bash
scl enable devtoolset-8 bash
python3 -m venv myvenv
source myvenv/bin/activate
pip3 install numba matplotlib scipy tqdm SALib tikzplotlib

git clone https://github.com/carlidel/c_henon_map.git
cd c_henon_map
pip3 install .
cd ..

mkdir data
mkdir img
python3 comparison.py
tar -czvf data.tar.gz img data
