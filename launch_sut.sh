#!/bin/bash

set -x
export LD_PRELOAD=${CONDA_PREFIX}/lib/libjemalloc.so
export MALLOC_CONF="oversize_threshold:1,background_thread:true,percpu_arena:percpu,metadata_thp:always,dirty_decay_ms:9000000000,muzzy_decay_ms:9000000000";

mode=${1:-"Offline"}
accuracy=${2:-""}

sut_dir=$(pwd)
executable=${sut_dir}/build/rnnt_inference
work_dir=${sut_dir}/mlperf-rnnt-librispeech
out_dir="${work_dir}/logs/${mode}"
mkdir -p ${out_dir} ${work_dir}

if [[ ${mode} == "Offline" ]]; then
  num_instance=1
  core_per_instance=56
  batch_size=32
elif [[ ${mode} == "Server" ]]; then
  num_instance=1
  core_per_instance=80
  batch_size=128
fi

config="--test_scenario=${mode} \
	--model_file=${work_dir}/rnnt_jit_quant.pt \
	--sample_file=${work_dir}/data_input_test.pt \
	--mlperf_config=${sut_dir}/inference/mlperf.conf \
	--user_config=${sut_dir}/configs/user.conf \
	--output_dir=${out_dir} \
	--inter_parallel=${num_instance} \
        --intra_parallel=${core_per_instance} \
	--batch_size=${batch_size} \
	${accuracy}"

${executable} ${config}

if [[ ${accuracy} == "--accuracy" ]]; then 
  python eval_accuracy.py \
    --log_path=${out_dir}/mlperf_log_accuracy.json \
    --manifest_path=${work_dir}/local_data/wav/dev-clean-wav.json 
fi

set +x

