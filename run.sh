#!/bin/bash

set -x

CONDA_ENV=${1:-'rnnt-infer'}
WORK_DIR=${2:-${PWD}/mlperf-rnnt-librispeech}
LOCAL_DATA_DIR=${WORK_DIR}/local_data
STAGE=${3:-2}

mkdir -p ${WORK_DIR}

if [[ ${STAGE} -le -1 ]]; then
  echo '==> Preparing env'
  source ./prepare_env.sh ${CONDA_ENV} ${PWD}
fi

conda activate ${CONDA_ENV}

if [[ ${STAGE} -le 0 ]]; then
  echo '==> Downloading model'
  wget https://zenodo.org/record/3662521/files/DistributedDataParallel_1576581068.9962234-epoch-100.pt?download=1 -O ${WORK_DIR}/rnnt.pt
fi

if [[ ${STAGE} -le 1 ]]; then
  echo '==> Downloading dataset'
  mkdir -p ${LOCAL_DATA_DIR}/LibriSpeech ${LOCAL_DATA_DIR}/raw
  python datasets/download_librispeech.py \
    --input_csv=configs/librispeech-inference.csv \
    --download_dir=${LOCAL_DATA_DIR}/LibriSpeech \
    --extract_dir=${LOCAL_DATA_DIR}/raw
fi

if [[ ${STAGE} -le 2 ]]; then
  echo '==> Pre-processing dataset'
  export PATH="${PWD}/third_party/bin/:${PATH}"
  python datasets/convert_librispeech.py \
    --input_dir=${LOCAL_DATA_DIR}/raw/dev-clean \
    --output_dir=${LOCAL_DATA_DIR}

  python datasets/convert_librispeech.py \
    --input_dir=${LOCAL_DATA_DIR}/raw/train-clean-100 \
    --output_dir=${LOCAL_DATA_DIR} \
    --output_list=configs/calibration_files.txt
fi

exit 78

# TODO: STAGE 3:do calibration
if [[ ${STAGE} -le 3 ]]; then
  echo '==> Calibrating'
  calibration.sh
fi

# TODO: STAGE 4:build model
if [[ ${STAGE} -le 4 ]]; then
  echo '==> Building model'
  build_model.sh 
fi

if [[ ${STAGE} -le 5 ]]; then
  for accuracy in "--accuracy" ""; do
    for scenario in Offline Server; do
      echo '==> Run RNN-T ${scenario} ${accuracy}'
      run.sh ${scenario} ${accuracy}
    done
  done
  wait
fi

set +x
