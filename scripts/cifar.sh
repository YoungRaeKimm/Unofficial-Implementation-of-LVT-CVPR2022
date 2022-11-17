#!/bin/bash

py3clean ./
CUDA_VISIBLE_DEVICES=0 python -B -m run.py \
                        --ILtype task \
                        --dataset cifar100 \
                        --split 10 \
                        --LRS False \
                        --memory_size 200 \
                        -everytest False 