#!/bin/bash

set -x 

: ${WORK_DIR=${1:-${PWD}/mlperf-rnnt-librispeech}}
: ${BATCH_SIZE=${2:-128}}
: ${LOG_LEVEL=${3:-10}}
: ${MODEL_PATH=${4:-${WORK_DIR}/rnnt.pt}}
: ${DEBUG:=false}
: ${MODE:=f32}
: ${JIT:=false}

export PYTHONPATH=${PWD}:${PYTHONPATH}
export LOG_LEVEL=${LOG_LEVEL}

[ ${MODE} == "fake_quant" ] && MODEL_PATH="${WORK_DIR}/rnnt_calib.pt"

SCRIPT_ARGS=" --scenario Offline"
SCRIPT_ARGS+=" --batch_size ${BATCH_SIZE}"
SCRIPT_ARGS+=" --model_path ${MODEL_PATH}"
SCRIPT_ARGS+=" --manifest_path ${WORK_DIR}/local_data/wav/dev-clean-wav.json"
SCRIPT_ARGS+=" --dataset_dir ${WORK_DIR}/dev-clean-input.pt"
SCRIPT_ARGS+=" --log_dir ${WORK_DIR}/logs/offline"
SCRIPT_ARGS+=" --mlperf_conf ${PWD}/configs/mlperf.conf"
SCRIPT_ARGS+=" --user_conf ${PWD}/configs/user.conf"
SCRIPT_ARGS+=" --accuracy"
[ ${MODE} != "f32" ] && SCRIPT_ARGS+=" --run_mode ${MODE} --calib_path ${PWD}/tests/calibration_result_nv.cache"
[ ${JIT} == true ] && SCRIPT_ARGS+=" --jit"
[ ${DEBUG} == "pdb" ] && EXEC_ARGS="ipdb3"
[ ${DEBUG} == "gdb" ] && EXEC_ARGS="gdb --args python"
[ ${DEBUG} == false ] && EXEC_ARGS="python -u"

${EXEC_ARGS} models/main.py ${SCRIPT_ARGS}

set +x

