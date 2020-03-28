import argparse
import random
import torch
import sys
import os

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

    # check_args
    # if params.debug_mode is False:
    #     check_params(params)

    # tag&save
    params.tags = [e for e in params.tags.split(',')] if params.tags is not None else ['test']

    if params.gpu != 'none':
        os.environ["CUDA_VISIBLE_DEVICES"] = params.gpu

    # random_seed
    torch.manual_seed(params.random_seed)
    torch.cuda.manual_seed(params.random_seed)
    torch.cuda.manual_seed_all(params.random_seed)
    random.seed(params.random_seed)

    if torch.cuda.is_available():
        params.device = 'cuda'
        params.gpu = list(range(len(params.gpu.split(','))))
        if params.gpu is not None:
            torch.cuda.set_device(params.gpu[0])

    params.weight_path = f'weight/{params.name}/'
    params.save_path = f'log/{params.name}.log'
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


base_args = parser.add_argument_group('Base args')
base_args.add_argument('--debug_mode', type=str2bool, default='1')
base_args.add_argument('--save_path')
base_args.add_argument('--tags')
base_args.add_argument('--project', type=str, default="DKT")
base_args.add_argument('--name', type=str, default="name")
base_args.add_argument('--device', type=str, default='cpu')
base_args.add_argument('--gpu', type=str, default='none')
base_args.add_argument('--num_workers', type=int, default=1)
base_args.add_argument('--base_path', type=str, default='/shared')
base_args.add_argument('--weight_path', type=str)
base_args.add_argument('--weight_num', type=str)
base_args.add_argument('--machine_name', type=str)

train_args = parser.add_argument_group('Train args')
train_args.add_argument('--random_seed', type=int, default=1)
train_args.add_argument('--num_epochs', type=int, default=10)
train_args.add_argument('--num_layers', type=int, default=1)
train_args.add_argument('--min_size', type=int, default=10)
train_args.add_argument('--train_batch', type=int, default=64)
train_args.add_argument('--test_batch', type=int, default=64)
train_args.add_argument('--lr', type=float, default=0.001)
train_args.add_argument('--seq_size', type=int, default=20)
train_args.add_argument('--sample_rate', type=float, default=0.5)
train_args.add_argument('--is_augmented', type=str2bool, default='1')
train_args.add_argument('--warm_up_step_count', type=int, default=4000)
train_args.add_argument('--num_layer', type=int, default=2)
train_args.add_argument('--d_model', type=int, default=256)
train_args.add_argument('--dropout', type=float, default=0.2)
train_args.add_argument('--is_warm_up', type=str2bool, default='1')
train_args.add_argument('--eval_steps', type=int, default=5)
train_args.add_argument('--max_grad_norm', type=float, default=20)

args = get_args()


if __name__ == '__main__':
    args = get_args()
    print_args(args, False)


