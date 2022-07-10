#

#!/usr/bin/env bash

DATASET=(assist2009 assist2015_q2a_k5 assist2015_q2a_k25 assist2015_q2a_k50)
# DATASET=ednet

model=(saint)
seed=(1)

# for t in ${atype[@]}; do
for m in ${model[@]}; do
    for d in ${DATASET[@]}; do
        args=(
            --dataset_name ${d}
            # --batch_size
            # --num_epochs
        )
        echo `PYTHONPATH=/home/tako/yoonjin/pykt-toolkit python wandb_${m}_train.py ${args[@]}`
    done
done