#

#!/usr/bin/env bash

dataset=(algebra2005)
embtype=(R_sinu_1 R_sinu_2 R_sinu_5 R_sinu_25)
seed=(42 224 3407)
model=(sakt)

for d in ${dataset[@]}; do
    for m in ${embtype[@]}; do
        for e in ${model[@]}; do
            for s in ${seed[@]}; do
                args=(
                    --dataset_name ${d}
                    --model_name ${e}
                    --emb_type ${m}
                    --seed ${s}
                    --gpu 1
                    # --use_wandb 0
                    # --fold 1
                )
                echo `PYTHONPATH=/workspace/pykt-toolkit python wandb_${e}_train.py ${args[@]}`
                # echo `PYTHONPATH=/home/tako/yoonjin/pykt-toolkit python wandb_${m}_train.py ${args[@]}`
            done
        done
    done
done
