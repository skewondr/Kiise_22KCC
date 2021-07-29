import argparse
import random
import torch
import sys
import os
import numpy as np

import time
from datetime import datetime

parser = argparse.ArgumentParser()


def get_run_script():
    run_script = 'python'
    for e in sys.argv:
        run_script += (' ' + e)

    return run_script


def str2bool(v):
    if v.lower() in ('yes', 'true', 't', 'y', '1'):
        return True
    elif v.lower() in ('no', 'false', 'f', 'n', '0'):
        return False
    else:
        raise argparse.ArgumentTypeError('Boolean value expected.')


def get_args():
    params = parser.parse_args()
    params.run_script = get_run_script()

    if params.gpu != 'none':
        os.environ["CUDA_VISIBLE_DEVICES"] = params.gpu

    # random_seed
    torch.manual_seed(params.random_seed)
    torch.cuda.manual_seed(params.random_seed)
    torch.cuda.manual_seed_all(params.random_seed)
    random.seed(params.random_seed)
    np.random.seed(params.random_seed)

    if torch.cuda.is_available():
        params.device = 'cuda'
        if params.gpu is not "none": # use only one gpu
            params.gpu = int(params.gpu)
            torch.cuda.set_device(params.gpu)
        else: # use multiple gpu
            num_gpus = torch.cuda.device_count()
            # if num_gpus > 1 :
            #     network = nn.DataParallel(network)

    params.weight_path = f'weight/{params.name}/'
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
base_args.add_argument('--name', type=str, default=f"best_{now_str}")
base_args.add_argument('--device', type=str, default='cpu')
base_args.add_argument('--gpu', type=str, default='none')
base_args.add_argument('--num_workers', type=int, default=8)
#base_args.add_argument('--base_path', type=str, default='/shared/benchmarks/')
base_args.add_argument('--weight_path', type=str, default=f"{now_str}")
base_args.add_argument('--dataset_name', type=str, default='ASSISTments2009', choices=dataset_list)

model_list = ['DKT', 'DKVMN', 'NPA', 'SAKT']

model_args = parser.add_argument_group('Model args')
model_args.add_argument('--model', type=str, default='DKT', choices=model_list)

# DKT, NPA, SAKT
model_args.add_argument('--num_layers', type=int, default=1)
model_args.add_argument('--hidden_dim', type=int, default=100)
model_args.add_argument('--input_dim', type=int, default=100)
model_args.add_argument('--dropout', type=float, default=0.2)

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
train_args.add_argument('--lr', type=float, default=0.001)
train_args.add_argument('--seq_size', type=int, default=200)
train_args.add_argument('--warm_up_step_count', type=int, default=4000)
train_args.add_argument('--eval_steps', type=int, default=5)
train_args.add_argument('--cross_validation', type=str2bool, default='0')

ARGS = get_args()


if __name__ == '__main__':
    ARGS = get_args()
    print_args(ARGS)


