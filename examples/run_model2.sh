#

#!/usr/bin/env bash

DATASET=assist2015

model=(dkt sakt gkt saint)
for m in ${model[@]}; do
    args=(
        --dataset_name $DATASET
        # --q2a ${k}
        # --batch_size
        # --num_epochs
    )
    echo `PYTHONPATH=/home/tako/yoonjin/pykt-toolkit python wandb_${m}_train.py ${args[@]}`
done

for k in ${k_emb[@]}; do
    args=(
        --dataset_name $DATASET
        # --r2a ${k}
        # --batch_size
        # --num_epochs
    )
    echo `PYTHONPATH=/home/tako/yoonjin/pykt-toolkit python wandb_saint_train.py ${args[@]}`
done