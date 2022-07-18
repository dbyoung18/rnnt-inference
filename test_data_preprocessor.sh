#!/bin/bash

set -x
export LD_PRELOAD=${CONDA_PREFIX}/lib/libjemalloc.so
export MALLOC_CONF="oversize_threshold:1,background_thread:true,percpu_arena:percpu,metadata_thp:always,dirty_decay_ms:9000000000,muzzy_decay_ms:9000000000";

: ${SCENARIO=${1:-"Offline"}}
: ${ACCURACY=${2:-""}}
: ${DEBUG:=false}
: ${PROFILE:=false}
: ${INTER:=1}
: ${INTRA:=56}
: ${BATCH_SIZE:=128}

SUT_DIR=$(pwd)
EXECUTABLE=${SUT_DIR}/build/rnnt_inference
WORK_DIR=${SUT_DIR}/mlperf-rnnt-librispeech
OUT_DIR="${WORK_DIR}/logs/${SCENARIO}"
mkdir -p ${OUT_DIR} ${WORK_DIR}

if [[ ${SCENARIO} == "Offline" ]]; then
  num_instance=${INTER}
  core_per_instance=${INTRA}
  batch_size=${BATCH_SIZE}
elif [[ ${SCENARIO} == "Server" ]]; then
  num_instance=1
  core_per_instance=56
  batch_size=128
fi

SCRIPT_ARGS=" --test_scenario=${SCENARIO}"
SCRIPT_ARGS+=" --model_file=${WORK_DIR}/rnnt_jit.pt"
SCRIPT_ARGS+=" --sample_file=${WORK_DIR}/dev-clean-npy.pt"
SCRIPT_ARGS+=" --preprocessor_file=${WORK_DIR}/audio_preprocessor_jit.pt"
SCRIPT_ARGS+=" --mlperf_config=${SUT_DIR}/inference/mlperf.conf"
SCRIPT_ARGS+=" --user_config=${SUT_DIR}/configs/user.conf"
SCRIPT_ARGS+=" --output_dir=${OUT_DIR}"
SCRIPT_ARGS+=" --inter_parallel=${num_instance}"
SCRIPT_ARGS+=" --intra_parallel=${core_per_instance}"
SCRIPT_ARGS+=" --batch_size=${batch_size}"
# SCRIPT_ARGS+=" --accuracy"

[ ${PROFILE} == true ] && SCRIPT_ARGS+=" --profiler"

[ ${DEBUG} == "gdb" ] && EXEC_ARGS="gdb --args"
[ ${DEBUG} == "lldb" ] && EXEC_ARGS="lldb --"

${EXEC_ARGS} ${EXECUTABLE} ${SCRIPT_ARGS}

if [[ ${ACCURACY} == "--accuracy" ]]; then
  python eval_accuracy.py \
    --log_path=${OUT_DIR}/mlperf_log_accuracy.json \
    --manifest_path=${WORK_DIR}/local_data/wav/dev-clean-wav.json
fi

set +x

