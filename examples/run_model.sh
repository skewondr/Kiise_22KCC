#

#!/usr/bin/env bash

dataset=(assist2009 ednet)
embtype=(qid Q_pretrain)
seed=(42 224 3407)
for d in ${dataset[@]}; do
    for m in ${embtype[@]}; do
        for s in ${seed[@]}; do
            args=(
                --dataset_name ${d}
                --model_name dkt
                --emb_type ${m}
                --seed ${s}
                --gpu 0
            )
            echo `PYTHONPATH=/workspace/pykt-toolkit python wandb_dkt_train.py ${args[@]}`
            # echo `PYTHONPATH=/home/tako/yoonjin/pykt-toolkit python wandb_${m}_train.py ${args[@]}`
        done
    done
done
