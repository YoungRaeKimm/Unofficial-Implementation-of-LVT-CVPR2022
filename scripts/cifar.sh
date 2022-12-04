#!/bin/bash

timestamp=$(date +%s)

                        # --resume True \
                        # --resume_task 1 \
                        # --resume_time 20221201_1356 \
py3clean ./
CUDA_VISIBLE_DEVICES=0 python3 -B ../run.py \
                        --ILtype class \
                        --dataset cifar100 \
                        --split 10 \
                        --alpha 2. \
                        --beta 2. \
                        --gamma 2. \
                        --memory_size 1000 \
                        --rt 1. \
                        --num_head 2 \
                        --hidden_dim 512 \
                        # -everytest False \
                        # > ../ckpt/logs/$(date -d "today" +"%Y%m%d_%H%M").txt