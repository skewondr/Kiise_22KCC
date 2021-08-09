import argparse
import random
import torch
import sys
import os
import shutil
import numpy as np

import time
from datetime import datetime

parser = argparse.ArgumentParser()

def script_path(script_file):
    return os.path.join(os.getcwd(),script_file) if script_file is not None else None 

def get_run():
    run = 'python'
    for e in sys.argv:
        run += (' ' + e)

    return run

def str2bool(v):
    if v.lower() in ('yes', 'true', 't', 'y', '1', 'True'):
        return True
    elif v.lower() in ('no', 'false', 'f', 'n', '0', 'False'):
        return False
    else:
        raise argparse.ArgumentTypeError('Boolean value expected.')


def get_args():
    params = parser.parse_args()
    params.run = get_run()
    params.run_script = script_path(params.run_script)

    params.ckpt_path = os.path.join(params.ckpt_path, params.ckpt_name) #./checkpoint/{ckpt_name}
    params.weight_path = os.path.join(params.weight_path, params.ckpt_name) #./checkpoint/{ckpt_name}
    if params.mode == 'train' and not params.ckpt_resume: 
        if os.path.exists(params.ckpt_path): #기존 폴더가 있으면 지우고 새로 생성함. 
            shutil.rmtree(params.ckpt_path)
        os.makedirs(params.ckpt_path, exist_ok=False)
    #params.weight_path = os.path.join(params.weight_path, params.weight_name) #./weight/{weight_name}
    os.makedirs(params.weight_path, exist_ok=True)

    return params


def print_args(params):
    info = '\n[args]________________________________________________\n'
    for sub_args in parser._action_groups:
        if sub_args.title in ['positional arguments', 'optional arguments']:
            continue
        size_sub = len(sub_args._group_actions)
        info += f'├─ {sub_args.title} ({size_sub})\n'
        for i, arg in enumerate(sub_args._group_actions):
            prefix = '└─' if i == size_sub-1 else '├─'
            info += f'│     {prefix} {arg.dest:20s}: {getattr(params, arg.dest)}\n'
    info += '└─────────────────────────────────────────────────────\n'
    print(info)

now = datetime.now()
now_str = f'{now.day:02}{now.hour:02}{now.minute:02}'

dataset_list = ['ASSISTments2009', 'ASSISTments2012', 'ASSISTments2015', 'ASSISTmentsChall',
                'STATICS', 'KDDCup', 'Junyi', 'EdNet-KT1']

base_args = parser.add_argument_group('Base args')

base_args.add_argument('--mode', type=str, default="train", help="train or eval")

base_args.add_argument('--device', type=str, default='cpu', help='automatically using GPU if you have cuda.')
base_args.add_argument('--gpu', type=str, default='none', help='using single gpu.')
base_args.add_argument('--num_workers', type=int, default=8)
#base_args.add_argument('--base_path', type=str, default='/shared/benchmarks/')

base_args.add_argument('--weight_path', type=str, default=f"./weight", help="saved model path")
#base_args.add_argument('--weight_name', type=str, default=f"best_{now_str}", help="saved model name")
base_args.add_argument('--ckpt_path', type=str, default="./checkpoint", help="checkpoint path")
base_args.add_argument('--ckpt_name', type=str, required=True, help="checkpoint name")
base_args.add_argument('--ckpt_resume', type=str2bool, default='0')
base_args.add_argument('--run_script', type=str, default="run_model.sh", help="Run script file path to log")

base_args.add_argument('--dataset_name', type=str, default='ASSISTments2009', choices=dataset_list)
base_args.add_argument('--get_user_ft', type=str2bool, default='0')
base_args.add_argument('--test_run', type=str2bool, default='0')
base_args.add_argument('--es_patience', type=int, default=10)

model_list = ['DKT', 'DKVMN', 'NPA', 'SAKT', 'KTM']

model_args = parser.add_argument_group('Model args')
model_args.add_argument('--model', type=str, default='DKT', choices=model_list)

# DKT, NPA, SAKT
model_args.add_argument('--num_layers', type=int, default=1)
model_args.add_argument('--hidden_dim', type=int, default=100)
model_args.add_argument('--input_dim', type=int, default=100)
model_args.add_argument('--dropout', type=float, default=0.2, help="probability of an element to be zeroed. Default: 0.5") 

# DKVMN
model_args.add_argument('--key_dim', type=int, default=100)
model_args.add_argument('--value_dim', type=int, default=100)
model_args.add_argument('--summary_dim', type=int, default=100)
model_args.add_argument('--concept_num', type=int, default=20)

# NPA
model_args.add_argument('--attention_dim', type=int, default=256)
model_args.add_argument('--fc_dim', type=int, default=512)

# SAKT
model_args.add_argument('--num_head', type=int, default=5)

train_args = parser.add_argument_group('Train args')
train_args.add_argument('--random_seed', type=int, default=1)
train_args.add_argument('--num_epochs', type=int, default=10)
train_args.add_argument('--train_batch', type=int, default=64)
train_args.add_argument('--test_batch', type=int, default=64)
train_args.add_argument('--lr', type=float, default=1e-3)
train_args.add_argument('--decay', type=float, default=2e-4)
train_args.add_argument('--eta_min', type=float, default=1e-4, help="Minimum learning rate for cosine annealing scheduler")
train_args.add_argument('--seq_size', type=int, default=200)
#train_args.add_argument('--warm_up_step_count', type=int, default=4000)
train_args.add_argument('--eval_steps', type=int, default=5)
train_args.add_argument('--cross_validation', type=str2bool, default='0')

ARGS = get_args()


if __name__ == '__main__':
    ARGS = get_args()
    print_args(ARGS)


