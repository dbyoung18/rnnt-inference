#!/bin/bash

set -x 

: ${WORK_DIR=${1:-${PWD}/mlperf-rnnt-librispeech}}
: ${BS=${2:-128}}
: ${LOG_LEVEL=${3:-10}}
: ${MODEL_PATH=${4:-${WORK_DIR}/rnnt.pt}}
: ${DEBUG:=false}
: ${WAV:=false}

export PYTHONPATH=${PWD}:${PWD}/models:${PYTHONPATH}
export RNNT_LOG_LEVEL=${LOG_LEVEL}

SCRIPT_ARGS=" --batch_size ${BS}"
SCRIPT_ARGS+=" --model_path ${MODEL_PATH}"
if [[ ${WAV} == true ]]; then
  SCRIPT_ARGS+=" --dataset_dir ${WORK_DIR}/train-clean-100-npy.pt --toml_path configs/rnnt.toml --enable_preprocess"
else
  SCRIPT_ARGS+=" --dataset_dir ${WORK_DIR}/train-clean-100-input.pt"
fi

[ ${DEBUG} == "pdb" ] && EXEC_ARGS="ipdb3"
[ ${DEBUG} == "gdb" ] && EXEC_ARGS="gdb --args python"
[ ${DEBUG} == "lldb" ] && EXEC_ARGS="lldb python --"
[ ${DEBUG} == false ] && EXEC_ARGS="python -u"

${EXEC_ARGS} models/calib_rnnt.py ${SCRIPT_ARGS}

set +x

