#

#!/usr/bin/env bash

# PYTHONPATH=/home/tako/yoonjin/pykt-toolkit python data_preprocess.py --dataset_name ednet 

dataset=(assist2009)
model=(dkt)
seed=(1 2 3)
for d in ${dataset[@]}; do
    for m in ${model[@]}; do
        for s in ${seed[@]}; do
            args=(
                --dataset_name ${d}
                --model_name ${m}
                --seed ${s}
                --seq_len 200
                --num_epochs 200
                --es_patience 10
                --fold 5
                # --gpu --> you can change
            )
            # echo `PYTHONPATH=/workspace/pykt-toolkit python wandb_saint_train.py ${args[@]}`
            echo `PYTHONPATH=/home/tako/yoonjin/pykt-toolkit python wandb_saint_train.py ${args[@]}`
        done
    done
done
