#!/bin/bash

# Create new env and activate it
conda install -y ninja cmake jemalloc inflect libffi pandas requests toml tqdm unidecode
conda install -c intel -y mkl mkl-include intel-openmp
conda install -c conda-forge -y llvm-openmp librosa
pip install -y sox
