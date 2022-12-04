#!/bin/bash

timestamp=$(date +%s)

                        # --resume True \
                        # --resume_task 1 \
                        # --resume_time 20221201_1356 \
py3clean ./
CUDA_VISIBLE_DEVICES=0 python3 -B ../run.py \
                        --ILtype task \
                        --dataset cifar100 \
                        --split 10 \
                        --alpha 1. \
                        --beta 1. \
                        --gamma 0.001 \
                        --memory_size 500 \
                        --rt 2. \
                        --num_head 2 \
                        --hidden_dim 512 \
                        # -everytest False \
                        # > ../ckpt/logs/$(date -d "today" +"%Y%m%d_%H%M").txt