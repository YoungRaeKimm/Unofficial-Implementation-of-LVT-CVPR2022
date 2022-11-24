#!/bin/bash

py3clean ./
CUDA_VISIBLE_DEVICES=0 python -B ../run.py \
                        --ILtype task \
                        --dataset cifar100 \
                        --split 10 \
                        --alpha 0.5 \
                        --beta 0.5 \
                        --gamma 0.5 \
                        --memory_size 200 \
                        -everytest False 