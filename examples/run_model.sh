#

#!/usr/bin/env bash

dataset=(assist2009 assist2015 ednet)
model=(emb)
for d in ${dataset[@]}; do
    for m in ${model[@]}; do
        args=(
            --dataset_name ${d}
            --model_name ${m}
        )
        echo `PYTHONPATH=/workspace/pykt-toolkit python wandb_${m}_train.py ${args[@]}`
        # echo `PYTHONPATH=/home/tako/yoonjin/pykt-toolkit python wandb_${m}_train.py ${args[@]}`
    done
done
