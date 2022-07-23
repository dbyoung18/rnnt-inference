#!/bin/bash

set -x 
export LD_PRELOAD=${CONDA_PREFIX}/lib/libjemalloc.so
export MALLOC_CONF="oversize_threshold:1,background_thread:true,percpu_arena:percpu,metadata_thp:always,dirty_decay_ms:9000000000,muzzy_decay_ms:9000000000";

: ${WORK_DIR=${1:-${PWD}/mlperf-rnnt-librispeech}}
: ${BATCH_SIZE=${2:-128}}
: ${LOG_LEVEL=${3:-10}}
: ${SCENARIO:=Offline}
: ${DEBUG:=false}
: ${MODE:=f32}
: ${WAV:=false}
: ${JIT:=false}

export PYTHONPATH=${PWD}:${PWD}/models/:${PYTHONPATH}
export RNNT_LOG_LEVEL=${LOG_LEVEL}

SCRIPT_ARGS=" --scenario ${SCENARIO}"
SCRIPT_ARGS+=" --batch_size ${BATCH_SIZE}"
SCRIPT_ARGS+=" --manifest_path ${WORK_DIR}/local_data/wav/dev-clean-wav.json"
SCRIPT_ARGS+=" --log_dir ${PWD}/logs/${SCENARIO}"
SCRIPT_ARGS+=" --mlperf_conf ${PWD}/configs/mlperf.conf"
SCRIPT_ARGS+=" --user_conf ${PWD}/configs/user.conf"
SCRIPT_ARGS+=" --split_fc1"
SCRIPT_ARGS+=" --accuracy"
if [[ ${MODE} == "fake_quant" ]]; then
  SCRIPT_ARGS+=" --run_mode ${MODE} --model_path ${WORK_DIR}/rnnt_calib.pt"
else
  SCRIPT_ARGS+=" --model_path ${WORK_DIR}/rnnt.pt"
fi

if [[ ${WAV} == true ]]; then
  SCRIPT_ARGS+=" --dataset_dir ${WORK_DIR}/dev-clean-npy.pt --toml_path configs/rnnt.toml --enable_preprocess"
else
  SCRIPT_ARGS+=" --dataset_dir ${WORK_DIR}/dev-clean-input.pt"
fi

[ ${JIT} == true ] && SCRIPT_ARGS+=" --jit"

[ ${DEBUG} == "pdb" ] && EXEC_ARGS="ipdb3"
[ ${DEBUG} == "gdb" ] && EXEC_ARGS="gdb --args python"
[ ${DEBUG} == "lldb" ] && EXEC_ARGS="lldb python --"
[ ${DEBUG} == false ] && EXEC_ARGS="python -u"

${EXEC_ARGS} models/main.py ${SCRIPT_ARGS}

set +x

