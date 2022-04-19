#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

#!/usr/bin/env bash

DATASET=EdNet-KT1
#DATASET=ASSISTments2009

model=(DKT)
prob=(0.04 0.08 0.16 0.32)
seed=(1 2 3)

# for t in ${atype[@]}; do
for m in ${model[@]}; do
    for p in ${prob[@]}; do
        for s in ${seed[@]}; do
            args=(
                #--mode 'eval'
                --result_path ./aug_results.txt #---------!
                --run_script ./model_args/Swap.sh #---------!
                --dataset_name $DATASET
                --seq_size 100 #---------!
                --sub_size 5
                --model ${m}
                --input_dim 200
                --hidden_dim 200
                --train_batch 2048
                --test_batch 2048
                --eval_steps 40000
                --ckpt_name Swap_${m}_${p}_${s} #---------!
                --lr 1e-3
                # --balance 1
                #######################################
                --aug_prob ${p} 
                --aug_type swapping #deletion / swapping / shuffling
                # --select_type ${t}
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