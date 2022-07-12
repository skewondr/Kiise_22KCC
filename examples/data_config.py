import os, sys
import argparse

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

def get_run():
    run = 'python'
    for e in sys.argv:
        run += (' ' + e)

    return run

def get_args():
    params = parser.parse_args()
    params.run = get_run()
    return params

parser = argparse.ArgumentParser()
data_args = parser.add_argument_group('Data preprocess args')
data_args.add_argument("--dataset_name", type=str, default="assist2009")
data_args.add_argument("--min_seq_len", type=int, default=3)
data_args.add_argument("--maxlen", type=int, default=100)
data_args.add_argument("--kfold", type=int, default=3)
data_args.add_argument("--subset", type=int, default=100)
ARGS = get_args()
print_args(ARGS)

if __name__ == "__main__":
    ARGS = get_args()
    print_args(ARGS)
