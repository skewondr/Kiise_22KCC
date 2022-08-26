#

#!/usr/bin/env bash

dataset=(ednet)
embtype=(R_quantized_1 R_sinusoid_2 R_sinusoid_5 R_sinusoid_25 R_add_1)
seed=(42 224 3407)
for d in ${dataset[@]}; do
    for m in ${embtype[@]}; do
        for s in ${seed[@]}; do
            args=(
                --dataset_name ${d}
                --model_name sakt
                --emb_type ${m}
                --seed ${s}
                --gpu 0
                # --use_wandb 0
                # --fold 1
            )
            echo `PYTHONPATH=/workspace/pykt-toolkit python wandb_sakt_train.py ${args[@]}`
            # echo `PYTHONPATH=/home/tako/yoonjin/pykt-toolkit python wandb_${m}_train.py ${args[@]}`
        done
    done
done
