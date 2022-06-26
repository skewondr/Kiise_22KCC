# Knowledge Tracing Models

*Implementations of various Knowledge Tracing models in [PyTorch](https://github.com/pytorch/pytorch)* 
*Code is based on [https://github.com/seewoo5/KT](https://github.com/seewoo5/KT)*

## Pre-processed Dataset
* dataset location : '../dataset/<dataset_name>/processed/1/sub<sub_size>/...'
* relevant files
```
util.py
dataset_user_sep.py
```

## Usage
* use '.sh' for convinience. 
```
bash run_model.sh
```
* or directly execution.
```
python main.py --num_workers=8 --gpu=0 --device=cuda --model=DKT --num_epochs=6 
--eval_steps=5000 --train_batch=2048 --test_batch=2048 --seq_size=200 
--input_dim=100 --hidden_dim=100 --name=ASSISTments2009_DKT_dim_100_100 
--dataset_name=ASSISTments2009 --cross_validation=1
```

Here are descriptions of arguments:

* `name`: name of the run. More precisely, the weight of the best model will be saved in the directory `weight/{ARGS.name}/`. 
* `gpu`: number(s) of gpu. 
* `device`: device. cpu, cuda, or others. 
* `base_path`: the path where datasets are located. 
* `num_workers`: number of workers for gpu training.
* `dataset_name`: the name of the benchmark dataset. Currently, ASSISTments2009, ASSISTments2015, ASSISTmentsChall, STATICS, Junyi, and EdNet-KT1 are available. 

* `model`: name of the model. DKT, DKVMN, or NPA. (SAKT is not available yet)
* `num_layers`: number of LSTM layers, for DKT and NPA. Set to be 1 as a default value. 
* `input_dim`: input embedding dimension of interactions, for DKT and NPA. <span style="color:red">(set to 200)</span>
* `hidden_dim`: hidden dimension of LSTM models, for DKT and NPA. <span style="color:red">(set to 200)</span>
* `key_dim`: dimension of key vectors of DKVMN.
* `value_dim`: dimension of value vectors of DKVMN.
* `summary_dim`: dimension of the last FC layer of DKVMN.
* `concept_num`: number of latent concepts, for DKVMN.
* `attention_dim`: dimension of the attention layer of NPA.
* `fc_dim`: largest dimension for the last FC layers of NPA.
* `num_head`: number of head of SAKT.
* `dropout`: dropout rate of the model. <span style="color:red">(set to 5)</span>

* `random_seed`: random seed for initialization, for reproducibility. Set to be 1 as default. 
* `num_epochs`: number of training epochs. <span style="color:red">(set to 20)</span>
* `eval_steps`: number of steps to evaluate trained model on validation set. The model weight with best performance will be saved. <span style="color:red">(set to 40000)</span>
* `train_batch`: batch size while training. <span style="color:red">(set to 2048)</span>
* `test_batch`: batch size while testing. <span style="color:red">(set to 2048)</span>
* `lr`: learning rate. 
* `es_patience`: early stopping patience <span style="color:red">(set to 3)</span>
* `warmup_step`: warmup step for Noam optimizer. 
* `seq_size`: length of interaction sequence to be feeded into models. The sequence whose length is shorter than `seq_size` will be padded. <span style="color:red">(set to 100)</span>
* `cross_validation`: if `cross_validation` is 0, then the model is trained & tested only on the first dataset. If `cross_validation` is 1, then the model is trained & tested on all 5 splits, and give average results (with standard deviation). 

## Common features
* All models are trained with Noam optimizer. 

## DKT (Deep Knowledge Tracing)
* Paper: https://web.stanford.edu/~cpiech/bio/papers/deepKnowledgeTracing.pdf
* Model: RNN, LSTM (only LSTM is implemented)
* GitHub: https://github.com/chrispiech/DeepKnowledgeTracing (Lua)
* Performances: 

| Dataset          | RMSE (%) | ACC (%) | AUC (%) | Time |
|------------------|-----|-----|-----|-----|
| EdNet-KT1(SUB 1)  | 58.80 ± 0.03 | 65.43 ± 0.04 | 70.20 ± 0.04 | 48sec | 
| EdNet-KT1(SUB 5)  | 57.41 ± 0.06 | 67.03 ± 0.08 | 72.65 ± 0.13 | 2min 41sec | 
| EdNet-KT1(SUB 50)  | 55.53 ± 0.01 | 69.16 ± 0.01 | 75.54 ± 0.04 | 13min 51sec | 

## SAKT (Self-Attentive Knowledge Tracing)
* Paper: https://files.eric.ed.gov/fulltext/ED599186.pdf
* Model: Transformer (1-layer, only encoder with subsequent mask)
* Github: https://github.com/shalini1194/SAKT (Tensorflow)
* Performances: 

| Dataset          | RMSE (%) | ACC (%) | AUC (%) | Time |
|------------------|-----|-----|-----|-----|
| EdNet-KT1(SUB 1)  | 59.36 ± 0.12 | 64.76 ± 0.14 | 68.77 ± 0.16 | 1min 29sec | 
| EdNet-KT1(SUB 5)  | 57.29 ± 0.04 | 67.13 ± 0.12 | 72.77 ± 0.10 | 12min | 
| EdNet-KT1(SUB 50)  | 55.35 ± 0.01 | 69.36 ± 0.02 | 75.82 ± 0.01 | 1h 4min | 



