#!/usr/bin/env bash

DATASET=EdNet-KT1
#DATASET=ASSISTments2009

MODEL=SAKT
#MODEL=KTM
#MODEL=SEQFM
#MODEL=DKT

args=(
    #--mode 'eval'
    --dataset_name $DATASET
    --seq_size 100
    --sub_size 5
    --model $MODEL
    #--alpha_model $A_MODEL
    --fm_hidden_dim 20
    --hidden_dim 200
    --train_batch 2048
    --test_batch 2048
    --eval_steps 40000
    --ckpt_name Testing #!!!!!!!!!!!!!SHOULD BE CHANGED!!!!!!!!!!!!!!!!
    --lr 1e-3
    --es_patience 30
    --num_head 5
    #--get_user_ft 1
    #--gpu 0
    #--cpu 1
    --num_workers 4
    #--random_seed
    #--num_epochs
    #--test_run 1
    #--ckpt_resume 1
)

python main.py "${args[@]}"
