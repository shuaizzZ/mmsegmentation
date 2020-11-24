#!/usr/bin/env bash

CONFIG=$1
GPUS=$2
PORT=${PORT:-29500}

PYTHONPATH="$(dirname $0)/..":$PYTHONPATH \
python launch.py --nproc_per_node=4 --master_port=123321 dis_train.py --launcher pytorch
