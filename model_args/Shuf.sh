#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

#!/usr/bin/env bash

DATASET=EdNet-KT1
#DATASET=ASSISTments2009

atype=(rnd)
model=(DKT)
ratio=(0.2 0.4 0.8)
seed=(1 2 3)

for t in ${atype[@]}; do
    for m in ${model[@]}; do
        for p in ${ratio[@]}; do
            for s in ${seed[@]}; do
                args=(
                    #--mode 'eval'
                    --result_path ./aug_results.txt #---------!
                    --run_script ./model_args/Shuf.sh #---------!
                    --dataset_name $DATASET
                    --seq_size 100 #---------!
                    --sub_size 5
                    --model ${m}
                    --input_dim 200
                    --hidden_dim 200
                    --train_batch 2048
                    --test_batch 2048
                    --eval_steps 40000
                    --ckpt_name Shuf_${t}_${m}_r${p}_${s} #---------!
                    --lr 1e-3
                    # --balance 1
                    #######################################
                    --aug_prob 0.32
                    --aug_ratio ${p}
                    --aug_type shuffling #deletion / swapping / shuffling
                    --select_type ${t}
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
done