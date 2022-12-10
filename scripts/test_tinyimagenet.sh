#!/bin/bash

py3clean ./
CUDA_VISIBLE_DEVICES=0 python3 -B ../run.py \
                        --test \
                        --ILtype task \
                        --dataset tinyimagenet200 \
                        --datapath /data/nahappy15/tinyimagenet200/ \
                        --num_head 4 \
                        --hidden_dim 512 \