#!/bin/bash

py3clean ./
CUDA_VISIBLE_DEVICES=0 python -B -m run.py \
                        --ILtype task \
                        --dataset tinyimagenet200 \
                        --split 10 \
                        --LRS True \
                        --memory_size 200 \
                        -everytest False 