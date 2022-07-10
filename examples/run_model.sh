#

#!/usr/bin/env bash

DATASET=assist2015

k_emb=(5 25 50)
for k in ${k_emb[@]}; do
    args=(
        --dataset_name $DATASET
        --q2a ${k}
        # --batch_size
        # --num_epochs
    )
    echo `PYTHONPATH=/home/tako/yoonjin/pykt-toolkit python wandb_saint_train.py ${args[@]}`
done

for k in ${k_emb[@]}; do
    args=(
        --dataset_name $DATASET
        --r2a ${k}
        # --batch_size
        # --num_epochs
    )
    echo `PYTHONPATH=/home/tako/yoonjin/pykt-toolkit python wandb_saint_train.py ${args[@]}`
done