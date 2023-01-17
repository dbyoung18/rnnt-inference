#!/bin/bash

set -ex
export LD_PRELOAD=${CONDA_PREFIX}/lib/libjemalloc.so
export MALLOC_CONF="oversize_threshold:1,background_thread:true,percpu_arena:percpu,metadata_thp:always,dirty_decay_ms:30000,muzzy_decay_ms:-1"

: ${PRO_BS:=4}
: ${PRO_INTER:=16}
: ${PRO_INTRA:=1}
: ${BS:=128}
: ${INTER:=28}
: ${INTRA:=4}
: ${LEN:=-1}
: ${RESPONSE:=-1}
: ${SCENARIO=${2:-"Offline"}}
: ${ACCURACY:=false}
: ${DEBUG:=false}
: ${MODE:=quant}
: ${WAV:=true}
: ${PROFILE:=-1}
: ${WARMUP:=-1}
: ${VERSION=${1:-"original"}}

SUT_DIR=$(pwd)
EXECUTABLE=${SUT_DIR}/build/rnnt_inference
WORK_DIR=${SUT_DIR}/mlperf-rnnt-librispeech
OUT_DIR="${SUT_DIR}/logs/${SCENARIO}_${VERSION}_${WAV}_PBS${PRO_BS}_BS${BS}_${PRO_INTER}_${PRO_INTRA}_${INTER}_${INTRA}_SL${LEN}"
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
SCRIPT_ARGS+=" --model_file=${WORK_DIR}/rnnt_${MODE}_jit.pt"
SCRIPT_ARGS+=" --mlperf_config=${SUT_DIR}/inference/mlperf.conf"
SCRIPT_ARGS+=" --user_config=${SUT_DIR}/configs/user.conf"
SCRIPT_ARGS+=" --output_dir=${OUT_DIR}"
SCRIPT_ARGS+=" --inter_parallel=${num_instance}"
SCRIPT_ARGS+=" --intra_parallel=${core_per_instance}"
SCRIPT_ARGS+=" --batch_size=${batch_size}"
SCRIPT_ARGS+=" --split_len=${LEN}"
SCRIPT_ARGS+=" --warmup_iter=${WARMUP}"

if [[ ${WAV} == true ]]; then
  SCRIPT_ARGS+=" --sample_file=${WORK_DIR}/dev-clean-npy.pt --processor_file=${WORK_DIR}/processor_jit.pt --processor"
else
  SCRIPT_ARGS+=" --sample_file=${WORK_DIR}/dev-clean-input.pt --processor_file=${WORK_DIR}/processor_jit.pt"
fi

if [[ ${SCENARIO} == "Server" ]]; then
  SCRIPT_ARGS+=" --pro_inter_parallel=${PRO_INTER} --pro_intra_parallel=${PRO_INTRA} --pro_batch_size=${PRO_BS} --response_size=${RESPONSE}"
fi
if [[ ${ACCURACY} == true ]]; then
  SCRIPT_ARGS+=" --accuracy"
else
  SCRIPT_ARGS+=" --profiler_iter=${PROFILE}"
fi

[ ${DEBUG} != false ]  && EXECUTABLE=${SUT_DIR}/build_dbg/rnnt_inference
[ ${DEBUG} == "gdb" ] && EXEC_ARGS="gdb --args"
[ ${DEBUG} == "lldb" ] && EXEC_ARGS="lldb --"
[ ${DEBUG} == "memcheck" ] && EXEC_ARGS="valgrind --leak-check=yes --gen-suppressions=all"

${EXEC_ARGS} ${EXECUTABLE} ${SCRIPT_ARGS}

if [[ ${ACCURACY} == true ]]; then
  export PYTHONPATH=${PWD}:${PWD}/models/:${PYTHONPATH}
  python -u eval_accuracy.py \
    --log_path=${OUT_DIR}/mlperf_log_accuracy.json \
    --manifest_path=${WORK_DIR}/local_data/wav/dev-clean-wav.json
fi

set +x
