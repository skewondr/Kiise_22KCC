#!/usr/bin/env bash

DATASET=EdNet-KT1
#DATASET=ASSISTments2009

MODEL=SAKT
#MODEL=KTM
#MODEL=SEQFM
# MODEL=DKT
seed=(1)
for s in ${seed[@]}; do
    args=(
        #--mode 'eval'
        --dataset_name $DATASET
        --seq_size 100
        --sub_size 5
        --model $MODEL
        --hidden_dim 200
        --train_batch 2048
        --test_batch 2048
        --eval_steps 40000
        --ckpt_name Testing #!!!!!!!!!!!!!aug, num, seg, seed!!!!!!!!!!!!!!!!
        --lr 1e-3
        --es_patience 5
        --num_head 5
        --random_seed ${s}
        --num_epochs 20
        #--test_run 1
        #--ckpt_resume 1 
        --gpu 0
    )
    echo `python main.py ${args[@]}`
done