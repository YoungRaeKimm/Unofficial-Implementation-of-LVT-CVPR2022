#!/bin/bash

timestamp=$(date +%s)

                        # --resume False \
                        # --resume_task 1 \
py3clean ./
CUDA_VISIBLE_DEVICES=0 python3 -B ../run.py \
                        --ILtype task \
                        --dataset cifar100 \
                        --split 10 \
                        --alpha 0.5 \
                        --beta 0.5 \
                        --gamma 0.5 \
                        --memory_size 500 \
                        # -everytest False \
                        # > ../ckpt/logs/$(date -d "today" +"%Y%m%d_%H%M").txt