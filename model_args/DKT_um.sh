#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

#!/usr/bin/env bash

DATASET=EdNet-KT1
#DATASET=ASSISTments2009

# MODEL=SAKT
MODEL=DKT
# MODEL=SAKT_LSTM
seed=(1 2 3)

for s in ${seed[@]}; do
    args=(
        #--mode 'eval'
        --dataset_name $DATASET
        --seq_size 100
        --sub_size 5
        --model $MODEL
        --input_dim 200
        --hidden_dim 200
        --train_batch 2048
        --test_batch 2048
        --eval_steps 40000
        --ckpt_name DKT_um_${s} #!!!!!!!!!!!!!seg, seed!!!!!!!!!!!!!!!!
        --qd 100
        --cd 50 
        --pd 50 
        --lr 1e-3
        # --balance 1
        # --bidirect 1
        --mask_prob 0.1
        --gumbel_temp 0.5 
        #######################################
        --g_use 1 #----------------------------gen 사용 여부
        --g_loss 0 #---------------------------gen 학습 여부
        --m_loss 1 #---------------------------kt model 학습 여부
        --g_train_flag 0 #---------------------train에서 gen 사용 여부
        --g_test_flag 0 #----------------------test에서 gen 사용 여부
        --mask_randomly 0 #--------------------input masking 시 랜덤성 여부 
        #######################################
        --loss_lambda 0.01
        --es_patience 3
        --num_head 5
        --random_seed ${s}
        --num_epochs 20
        #--test_run 1
        #--ckpt_resume 1 
        --gpu 0
    )
    echo `python main.py ${args[@]}`
done
