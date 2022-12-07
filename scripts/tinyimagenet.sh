#!/bin/bash

timestamp=$(date +%s)

py3clean ./
CUDA_VISIBLE_DEVICES=0 python3 -B ../run.py \
                        --ILtype task \
                        --dataset tinyimagenet200 \
                        --datapath /data/nahappy15/tinyimagenet200/ \
                        --split 10 \
                        --alpha 1. \
                        --beta 1. \
                        --gamma 0.5 \
                        --memory_size 1000 \
                        --rt 10. \
                        --num_head 4 \
                        --hidden_dim 512 \