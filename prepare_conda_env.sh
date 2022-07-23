#!/bin/bash

# Create new env and activate it
conda install -y python==3.8
conda install -y ninja
conda install -y cmake
conda install -c intel -y mkl
conda install -c intel -y mkl-include
conda install -c intel -y intel-openmp
conda install -c intel -y llvm-openmp
conda install -y jemalloc
conda install -y inflect
conda install -y libffi
conda install -c conda-forge -y librosa
conda install -y pandas
conda install -y requests
conda install -y toml
conda install -y tqdm
conda install -y unidecode
