#

#!/usr/bin/env bash

dataset=(assist2009 ednet algebra2005)
embtype=(1 2)
seed=(42)
gap=(R_sinu_0 R_sinu_10 R_sinu_50 R_sinu_100 R_sinu_150 R_sinu_200)
model=(dkt dkvmn)

for d in ${dataset[@]}; do
    for t in ${model[@]}; do
        for m in ${embtype[@]}; do
            for e in ${gap[@]}; do
                for s in ${seed[@]}; do
                    args=(
                        --dataset_name ${d}
                        --model_name ${t}
                        --emb_type ${e}_${m}
                        --seed ${s}
                        --gpu 0
                        # --use_wandb 0
                        # --fold 1
                    )
                    echo `PYTHONPATH=/workspace/pykt-toolkit python wandb_${t}_train.py ${args[@]}`
                    # echo `PYTHONPATH=/home/tako/yoonjin/pykt-toolkit python wandb_${m}_train.py ${args[@]}`
                done
            done
        done
    done
done
