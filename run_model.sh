#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

#!/usr/bin/env bash

DATASET=EdNet-KT1
#DATASET=ASSISTments2009

model=(DKT)
sub=(1)
seed=(1)

# for t in ${atype[@]}; do
for m in ${model[@]}; do
    for p in ${sub[@]}; do
        for s in ${seed[@]}; do
            args=(
                #--mode 'eval'
                --result_path ./emb_results.txt #---------!
                --run_script run_model.sh #---------!
                --dataset_name $DATASET
                --seq_size 100 #---------!
                --sub_size ${p}
                --model ${m}
                --input_dim 200
                --hidden_dim 200
                --train_batch 2048
                --test_batch 2048
                --eval_steps 40000
                --ckpt_name Testing #---------!
                --lr 1e-3
                # --balance 1
                #######################################
                --qd 100
                --cd 100
                --pd 100
                --emb_type origin 
                #######################################
                --es_patience 3
                --num_head 5
                --random_seed ${s}
                --num_epochs 20
                #--test_run 1
                #--ckpt_resume 1 
                --gpu 0 #---------!
            )
            echo `python main.py ${args[@]}`
        done
    done
done
# done