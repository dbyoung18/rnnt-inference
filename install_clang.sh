#!/bin/bash
set -x

# install dependency
conda install -c anaconda ncurses
# set link path
export LD_LIBRARY_PATH=${CONDA_PREFIX}/lib:${LD_LIBRARY_PATH}
# build from source
git clone https://github.com/llvm/llvm-project.git
pushd llvm-project
mkdir build
pushd build
conda install -c anaconda ncurses
cmake ../llvm -GNinja -DLLVM_ENABLE_PROJECTS="clang;lldb" -DLLVM_ENABLE_RUNTIMES="libcxx;libcxxabi;openmp" -DCMAKE_INSTALL_PREFIX=$CONDA_PREFIX -DCMAKE_SHARED_LINKER_FLAGS="-L$CONDA_PREFIX -Wl,-rpath,$CONDA_PREFIX" -DCMAKE_BUILD_TYPE=Release 
ninja
ninja install
popd
popd

set +x
