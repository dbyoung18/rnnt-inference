#export MKL_DPCPP_ROOT=/opt/intel/oneapi/mkl/latest
#export LD_LIBRARY_PATH=${MKL_DPCPP_ROOT}/lib:${MKL_DPCPP_ROOT}/lib64/:${MKL_DPCPP_ROOT}/lib/intel64:${LD_LIBRARY_PATH}
#export LIBRARY_PATH=${MKL_DPCPP_ROOT}/lib:${MKL_DPCPP_ROOT}/lib64:${MKL_DPCPP_ROOT}/lib/intel64:${LIBRARY_PATH}
export CMAKE_PREFIX_PATH=${CONDA_PREFIX:-"$(dirname $(which conda))/../"}
export CMAKE_LIBRARY_PATH=${CMAKE_PREFIX_PATH}/lib
export CMAKE_INCLUDE_PATH=${CMAKE_PREFIX_PATH}/include
