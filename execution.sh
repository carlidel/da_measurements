#!/bin/bash
python -m virtualenv myvenv
source myvenv/bin/activate
pip install numba matplotlib scipy tqdm

git clone https://github.com/carlidel/c_henon_map.git
cd c_henon_map
pip install .
cd ..

mkdir data
mkdir img
python comparison.py
tar -czf data.tar.gz ./img ./data