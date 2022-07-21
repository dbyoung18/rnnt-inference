#!/bin/bash

set -x 

: ${WORK_DIR=${1:-${PWD}/mlperf-rnnt-librispeech}}
: ${BATCH_SIZE=${2:-128}}
: ${LOG_LEVEL=${3:-10}}
: ${MODEL_PATH=${4:-${WORK_DIR}/rnnt.pt}}
: ${DEBUG:=false}
: ${MODE:=""}
: ${JIT:=false}
: ${WAV:=false}


export PYTHONPATH=${PWD}:${PYTHONPATH}
export LOG_LEVEL=${LOG_LEVEL}

[ ${MODE} == "fake_quant" ] && MODEL_PATH="${WORK_DIR}/rnnt_calib.pt"
[ ${MODE} == "quant" ] && MODEL_PATH="${WORK_DIR}/rnnt_quant.pt"

SCRIPT_ARGS=" --scenario Offline"
SCRIPT_ARGS+=" --batch_size ${BATCH_SIZE}"
SCRIPT_ARGS+=" --model_path ${MODEL_PATH}"
SCRIPT_ARGS+=" --manifest_path ${WORK_DIR}/local_data/wav/dev-clean-wav.json"
SCRIPT_ARGS+=" --log_dir ${PWD}/logs/offline"
SCRIPT_ARGS+=" --mlperf_conf ${PWD}/configs/mlperf.conf"
SCRIPT_ARGS+=" --user_conf ${PWD}/configs/user.conf"
SCRIPT_ARGS+=" --toml_path ${PWD}/configs/rnnt.toml"
SCRIPT_ARGS+=" --split_fc1"
SCRIPT_ARGS+=" --accuracy"
[ ${WAV} == false ] && SCRIPT_ARGS+=" --dataset_dir ${WORK_DIR}/dev-clean-input.pt"
[ ${WAV} == true ] && SCRIPT_ARGS+=" --dataset_dir ${WORK_DIR}/dev-clean-npy.pt --enable_preprocess"
[ ${MODE} != "" ] && SCRIPT_ARGS+=" --run_mode ${MODE}"
[ ${JIT} == true ] && SCRIPT_ARGS+=" --jit"
[ ${DEBUG} == "pdb" ] && EXEC_ARGS="ipdb3"
[ ${DEBUG} == "gdb" ] && EXEC_ARGS="gdb --args python"
[ ${DEBUG} == false ] && EXEC_ARGS="python -u"

${EXEC_ARGS} models/main.py ${SCRIPT_ARGS}

set +x

