#!/bin/bash

set -x

CONDA_ENV=${1:-'rnnt-infer'}
HOME_DIR=${2:-${PWD}}

echo '==> Creating conda env'
conda env create -n ${CONDA_ENV} -f environment.yml --force -v
conda activate ${CONDA_ENV}

pushd ${HOME_DIR}

echo '==> Building mlperf-loadgen'
git clone --recurse-submodules https://github.com/mlcommons/inference.git
git checkout r2.1
pushd inference
git checkout master
git submodule sync
git submodule update --init --recursive
pushd loadgen
CFLAGS="-std=c++14" python setup.py install
popd
popd

echo '==> Building third-party'
third_party_dir=${HOME_DIR}/third_party
mkdir -p ${third_party_dir}
pushd ${third_party_dir}

wget https://ftp.osuosl.org/pub/xiph/releases/flac/flac-1.3.2.tar.xz -O flac-1.3.2.tar.xz
tar xf flac-1.3.2.tar.xz
pushd flac-1.3.2
./configure --prefix=${third_party_dir} && make && make install
popd

wget https://sourceforge.net/projects/sox/files/sox/14.4.2/sox-14.4.2.tar.gz -O sox-14.4.2.tar.gz
tar zxf sox-14.4.2.tar.gz
pushd sox-14.4.2
LDFLAGS="-L${third_party_dir}/lib" CFLAGS="-I${third_party_dir}/include" ./configure --prefix=${third_party_dir} --with-flac && make && make install
popd
popd

echo '==> Building pytorch'
git clone https://github.com/pytorch/pytorch.git
pushd pytorch
git checkout v1.12.0
git apply ../patches/pytorch_official_1_12.patch
git submodule sync
git submodule update --init --recursive
pip install -r requirements.txt
USE_CUDA=OFF python -m pip install -e .
popd

echo '==> Building mlperf_plugins, C++ loadgen & SUT'
rm ${CONDA_PREFIX}/lib/cmake/mkl/*
git submodule sync
git submodule update --init --recursive
mkdir build
pushd build
# cmake -DBUILD_TPPS_INTREE=ON -DCMAKE_PREFIX_PATH="$(dirname $(python3 -c 'import torch; print(torch.__file__)'))" -GNinja -DUSERCP=ON -DCMAKE_BUILD_TYPE=Release ..
cmake -DCMAKE_CXX_COMPILER=clang++ -DCMAKE_C_COMPILER=clang -DBUILD_TPPS_INTREE=ON -DCMAKE_PREFIX_PATH="$(dirname $(python3 -c 'import torch; print(torch.__file__)'));../cmake/Modules" -GNinja -DUSERCP=ON -DCMAKE_BUILD_TYPE=Release ..
ninja
popd

set +x

