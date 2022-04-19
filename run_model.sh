#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

#!/usr/bin/env bash

DATASET=EdNet-KT1
#DATASET=ASSISTments2009

# MODEL=SAKT
MODEL=DKT
# MODEL=SAKT_LSTM
seed=(1)

for s in ${seed[@]}; do
    args=(
        #--mode 'eval'
        --dataset_name $DATASET
        --seq_size 15
        --sub_size 5
        --model $MODEL
        --input_dim 200
        --hidden_dim 200
        --train_batch 2048
        --test_batch 2048
        --eval_steps 40000
        --ckpt_name Testing
        --lr 1e-3
        # --balance 1
        #######################################
        --aug_prob 0.2
        --aug_type shuffling
        --select_type gcr
        #######################################
        --es_patience 3
        --num_head 5
        --random_seed ${s}
        --num_epochs 20
        #--test_run 1
        #--ckpt_resume 1 
        --gpu 1
    )
    echo `python main.py ${args[@]}`
done