#!/bin/bash

set -x
export LD_PRELOAD=${CONDA_PREFIX}/lib/libjemalloc.so
export MALLOC_CONF="oversize_threshold:1,background_thread:true,percpu_arena:percpu,metadata_thp:always,dirty_decay_ms:9000000000,muzzy_decay_ms:9000000000";

: ${BS=${1:-128}}
: ${INTER:=2}
: ${INTRA:=56}
: ${SCENARIO=${2:-"Offline"}}
: ${ACCURACY:=false}
: ${PROFILE:=false}
: ${DEBUG:=false}
: ${WAV:=false}
: ${HT:=true}
: ${COUNT:=}

SUT_DIR=$(pwd)
EXECUTABLE=${SUT_DIR}/build/rnnt_inference
WORK_DIR=${SUT_DIR}/mlperf-rnnt-librispeech
OUT_DIR="${PWD}/logs/${SCENARIO}"
mkdir -p ${OUT_DIR} ${WORK_DIR}

if [[ ${SCENARIO} == "Offline" ]]; then
  num_instance=${INTER}
  core_per_instance=${INTRA}
  batch_size=${BS}
elif [[ ${SCENARIO} == "Server" ]]; then
  num_instance=${INTER}
  core_per_instance=${INTRA}
  batch_size=${BS}
fi

SCRIPT_ARGS=" --test_scenario=${SCENARIO}"
SCRIPT_ARGS+=" --model_file=${WORK_DIR}/rnnt_quant_jit_no_cf.pt"
SCRIPT_ARGS+=" --mlperf_config=${SUT_DIR}/inference/mlperf.conf"
SCRIPT_ARGS+=" --user_config=${SUT_DIR}/configs/user.conf"
SCRIPT_ARGS+=" --output_dir=${OUT_DIR}"
SCRIPT_ARGS+=" --inter_parallel=${num_instance}"
SCRIPT_ARGS+=" --intra_parallel=${core_per_instance}"
SCRIPT_ARGS+=" --batch_size=${batch_size}"

if [[ ${WAV} == true ]]; then
  SCRIPT_ARGS+=" --sample_file=${WORK_DIR}/dev-clean-npy.pt --preprocessor_file=${WORK_DIR}/preprocessor_jit.pt --preprocessor"
else
  SCRIPT_ARGS+=" --sample_file=${WORK_DIR}/dev-clean-input.pt --preprocessor_file=${WORK_DIR}/preprocessor_jit.pt"
fi

[ ${HT} == false ]  && SCRIPT_ARGS+=" --disable-hyperthreading"
[ ${PROFILE} == true ] && SCRIPT_ARGS+=" --profiler"
[ ${ACCURACY} == true ] && SCRIPT_ARGS+=" --accuracy"
if [[ ${ACCURACY} != true && ${COUNT} != "" ]]; then
  SCRIPT_ARGS+=" --perf_count ${COUNT}"
fi

[ ${DEBUG} == "gdb" ] && EXEC_ARGS="gdb --args"
[ ${DEBUG} == "lldb" ] && EXEC_ARGS="lldb --"

${EXEC_ARGS} ${EXECUTABLE} ${SCRIPT_ARGS}

if [[ ${ACCURACY} == true ]]; then
  python -u eval_accuracy.py \
    --log_path=${OUT_DIR}/mlperf_log_accuracy.json \
    --manifest_path=${WORK_DIR}/local_data/wav/dev-clean-wav.json
fi

set +x

