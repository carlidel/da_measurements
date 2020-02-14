#!/bin/bash
python -m virtualenv myvenv
source myvenv/bin/activate
pip install numba matplotlib scipy tqdm
mkdir full_data
mkdir img
python Full_comparison_gpu.py
tar -czf data.tar.gz ./img ./full_data