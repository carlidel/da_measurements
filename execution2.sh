#!/bin/bash
python3 -m venv myvenv
source myvenv/bin/activate
pip3 install numba matplotlib scipy tqdm SALib tikzplotlib

git clone https://github.com/carlidel/c_henon_map.git
cd c_henon_map
pip3 install .
cd ..

mkdir data
mkdir img
python3 part2.py
tar -czvf data2.tar.gz img data
split -b 500M data2.tar.gz "data2.tar.gz.part"
python3 flatter.py