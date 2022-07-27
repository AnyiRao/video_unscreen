#!/bin/bash

script=$1
src=$2
gpu_id=$3
PY_ARGS=${@:4}
echo unscreen video ${src} on gpuid ${gpu_id}

if [ -z "${gpu_id}" ]
then 
    echo "Devices not set. Use default the first 0 card"
    gpu_id="0"
fi
echo "Devices set to ${gpu_id}"

export PYTHONPATH=./
CUDA_VISIBLE_DEVICES=${gpu_id} python tools/unscreen/${script}.py --video_id ${src} ${PY_ARGS}
echo finished video ${src} on gpuid ${gpu_id}