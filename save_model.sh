#!/bin/bash

set -x 

: ${WORK_DIR=${1:-${PWD}/mlperf-rnnt-librispeech}}
: ${BATCH_SIZE=${2:-128}}
: ${LOG_LEVEL=${3:-10}}
: ${MODEL_PATH=${4:-${WORK_DIR}/rnnt_calib.pt}}
: ${DEBUG:=false}
: ${MODE:=f32}
: ${JIT:=false}

export PYTHONPATH=${PWD}:${PYTHONPATH}
export LOG_LEVEL=${LOG_LEVEL}

SCRIPT_ARGS=" --batch_size ${BATCH_SIZE}"
SCRIPT_ARGS+=" --model_path ${MODEL_PATH}"
SCRIPT_ARGS+=" --dataset_dir ${WORK_DIR}/train-clean-100-input.pt"
SCRIPT_ARGS+=" --split_fc1"
[ ${JIT} == true ] && SCRIPT_ARGS+=" --jit"
[ ${DEBUG} == "pdb" ] && EXEC_ARGS="ipdb3"
[ ${DEBUG} == "gdb" ] && EXEC_ARGS="gdb --args python"
[ ${DEBUG} == false ] && EXEC_ARGS="python -u"

${EXEC_ARGS} models/save_rnnt.py ${SCRIPT_ARGS}

set +x

