#!/bin/bash

CUDA_VISIBLE_DEVICES=0 python main.py --fp16 --task_name mnli-m \
                                      --seed 123       \
                                      --algorithm ffa  \
                                      --split_coef 0.1 \
                                      --n_clients 10   \

CUDA_VISIBLE_DEVICES=0 python main.py --fp16 --task_name qqp \
                                      --seed 123       \
                                      --algorithm ffa  \
                                      --split_coef 0.1 \
                                      --n_clients 10   \    

CUDA_VISIBLE_DEVICES=0 python main.py --fp16 --task_name sst2 \
                                      --seed 123       \
                                      --algorithm ffa  \
                                      --split_coef 0.1 \
                                      --n_clients 10   \

CUDA_VISIBLE_DEVICES=0 python main.py --fp16 --task_name qnli \
                                      --seed 123       \
                                      --algorithm ffa  \
                                      --split_coef 0.1 \
                                      --n_clients 10   \

CUDA_VISIBLE_DEVICES=0 python main.py --fp16 --task_name rte \
                                      --seed 123       \
                                      --algorithm ffa  \
                                      --split_coef 0.1 \
                                      --n_clients 10   \ 
