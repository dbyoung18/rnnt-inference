#!/bin/bash

set -x 
export LD_PRELOAD=${CONDA_PREFIX}/lib/libjemalloc.so
export MALLOC_CONF="oversize_threshold:1,background_thread:true,percpu_arena:percpu,metadata_thp:always,dirty_decay_ms:9000000000,muzzy_decay_ms:9000000000";

: ${WORK_DIR=${1:-${PWD}/mlperf-rnnt-librispeech}}
: ${BATCH_SIZE=${2:-128}}
: ${LOG_LEVEL=${3:-10}}
: ${MODEL_PATH=${4:-${WORK_DIR}/rnnt.pt}}
: ${DEBUG:=false}
: ${MODE:=f32}
: ${JIT:=false}
: ${WAV:=false}

export PYTHONPATH=${PWD}:${PYTHONPATH}
export LOG_LEVEL=${LOG_LEVEL}

DATASET_DIR=${WORK_DIR}/dev-clean-input.pt
[ ${WAV} == true ] & DATASET_DIR=${WORK_DIR}/dev-clean-npy.pt

SCRIPT_ARGS=" --scenario Offline"
SCRIPT_ARGS+=" --batch_size ${BATCH_SIZE}"
SCRIPT_ARGS+=" --model_path ${MODEL_PATH}"
SCRIPT_ARGS+=" --manifest_path ${WORK_DIR}/local_data/wav/dev-clean-wav.json"
SCRIPT_ARGS+=" --dataset_dir ${DATASET_DIR}"
SCRIPT_ARGS+=" --log_dir ${PWD}/logs/offline"
SCRIPT_ARGS+=" --mlperf_conf ${PWD}/configs/mlperf.conf"
SCRIPT_ARGS+=" --user_conf ${PWD}/configs/user.conf"
SCRIPT_ARGS+=" --toml_path ${PWD}/configs/rnnt.toml"
SCRIPT_ARGS+=" --split_fc1"
SCRIPT_ARGS+=" --enable_preprocess"
SCRIPT_ARGS+=" --accuracy"
[ ${JIT} == true ] && SCRIPT_ARGS+=" --jit"
[ ${DEBUG} == "pdb" ] && EXEC_ARGS="ipdb3"
[ ${DEBUG} == "gdb" ] && EXEC_ARGS="gdb --args python"
[ ${DEBUG} == "lldb" ] && EXEC_ARGS="lldb python --"
[ ${DEBUG} == false ] && EXEC_ARGS="python -u"

${EXEC_ARGS} models/main.py ${SCRIPT_ARGS}

set +x

