#!/bin/bash

set -x 

: ${WORK_DIR=${1:-${PWD}/mlperf-rnnt-librispeech}}
: ${BATCH_SIZE=${2:-128}}
: ${LOG_LEVEL=${3:-10}}
: ${MODEL_PATH=${4:-${WORK_DIR}/rnnt_calib.pt}}
: ${MODE:=quant}
: ${WAV:=true}
: ${SAVE_JIT:=true}
: ${DEBUG:=false}

export PYTHONPATH=${PWD}:${PWD}/models/:${PYTHONPATH}
export RNNT_LOG_LEVEL=${LOG_LEVEL}

SCRIPT_ARGS=" --batch_size ${BATCH_SIZE}"
SCRIPT_ARGS+=" --model_path ${MODEL_PATH}"
SCRIPT_ARGS+=" --run_mode ${MODE}"
SCRIPT_ARGS+=" --manifest_path ${WORK_DIR}/local_data/wav/dev-clean-wav.json"
if [[ ${WAV} == true ]]; then
  SCRIPT_ARGS+=" --calib_dataset_dir ${WORK_DIR}/dev-clean-npy.pt --toml_path configs/rnnt.toml --enable_preprocess"
else
  SCRIPT_ARGS+=" --calib_dataset_dir ${WORK_DIR}/dev-clean-input.pt"
fi

[ ${SAVE_JIT} == true ] && SCRIPT_ARGS+=" --save_jit"

[ ${DEBUG} == "pdb" ] && EXEC_ARGS="ipdb3"
[ ${DEBUG} == "gdb" ] && EXEC_ARGS="gdb --args python"
[ ${DEBUG} == "lldb" ] && EXEC_ARGS="lldb python --"
[ ${DEBUG} == false ] && EXEC_ARGS="python -u"

${EXEC_ARGS} models/main.py ${SCRIPT_ARGS}

set +x

