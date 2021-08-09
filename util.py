import csv
import glob
import os
import pickle
from tqdm import tqdm 
import numpy as np
from config import ARGS

import random
from typing import Dict, Optional, Tuple, Union

import re
import numpy as np
import torch
import torch.nn as nn
from torch.cuda.amp import GradScaler
from torch.optim import Optimizer, optimizer
from torch.optim.lr_scheduler import _LRScheduler

TModel = Union[nn.DataParallel, nn.Module]

def create_full_path(user_base_path, user_path):
    u0 = user_path[0]
    u1 = user_path[1]
    u2 = user_path[2]
    u3 = user_path[3]
    return f'{user_base_path}/{u0}/{u1}/{u2}/{u3}/{user_path}'


def get_qid_to_embed_id(dict_path):
    d = {}
    with open(dict_path, 'r') as f:
        lines = f.readlines()
        for line in lines:
            line = line.rstrip().split(',')
            d[int(line[0])] = int(line[1])
    return d


def get_sample_info(user_base_path, data_path):
    # for modified_AAAI20 data
    sample_infos = []
    with open(data_path, 'r') as f:
        lines = f.readlines()
        num_of_users = len(lines)
        for user_path in lines:
            user_path = user_path.rstrip()
            user_full_path = create_full_path(user_base_path, user_path)
            with open(user_full_path, 'r', encoding='ISO-8859-1') as f:
                lines = f.readlines()
                num_of_interactions = len(lines)
                for target_index in range(num_of_interactions):
                    sample_infos.append([user_path, target_index])

    return sample_infos, num_of_users

# Do not use this anymore
def get_data_tl(data_path):
    # for triple line format data
    sample_data = []
    with open(data_path, 'r') as f:
        lines = f.readlines()
        num_of_users = len(lines) // 3
        for i in range(num_of_users):
            user_interaction_len = int(lines[3*i].strip())
            qid_list = list(map(int, lines[3*i+1].split(',')))
            is_correct_list = list(map(int, lines[3*i+2].split(',')))
            assert user_interaction_len == len(qid_list) == len(is_correct_list), 'length is different'

            for j in range(user_interaction_len):
                sample_data.append((qid_list[:j+1], is_correct_list[:j+1]))

    return sample_data, num_of_users


def get_data_infos(user_base_path, i, mode): #question
    data_path = f"{user_base_path}/{i}/{mode}/"
    #sample_data_name = f"{user_base_path}/{i}/{mode}_{ARGS.dataset_name}_{ARGS.seq_size}_data.npz"
    sample_data_name = f"{user_base_path}/{i}/{mode}_{ARGS.dataset_name}_{ARGS.seq_size}_sample.pickle"

    # get list of all files
    user_path_list = os.listdir(data_path)
    num_of_users = len(user_path_list)

    if os.path.isfile(sample_data_name):
        print(f"Loading {sample_data_name}...")
        with open(sample_data_name, 'rb') as f: 
            sample_infos = pickle.load(f)
    else:
        # almost same as get_sample_info
        # for user separated format data
      
        sample_infos = []
        max_id = 0
        for idx, user_path in enumerate(tqdm(user_path_list, total=num_of_users, ncols=100)):
            user_id = user_path.split('/')[-1]
            user_id = int(re.sub(r'[^0-9]', '', user_id))
            if user_id>max_id: max_id = user_id 
            with open(data_path + user_path, 'rb') as f:
                lines = f.readlines()
                lines = lines[1:]  # header exists
                num_of_interactions = len(lines) # sequence length 
                if mode != 'val':
                    for end_index in range(5,num_of_interactions):
                        sample_infos.append((data_path + user_path,end_index))
                else:
                    sample_infos.append((data_path + user_path, num_of_interactions-1))

            #if idx > 100 : break
            
        with open(sample_data_name, 'wb') as f: pickle.dump(sample_infos, f)
    return sample_infos, num_of_users

def save_checkpoint(
    ckpt_path: str,
    model: TModel,
    epoch: int,
    optim: Optional[Optimizer] = None,
    scheduler: Optional[_LRScheduler] = None,
    other_states: dict = {},
) -> None:
    if isinstance(model, nn.DataParallel): 
        model = model.module
   
    state = {"net": model.state_dict()}
    state["epoch"] = epoch

    state["rng_state"] = ( #the random number generator state
        torch.get_rng_state(),
        np.random.get_state(),
        random.getstate(),
    )

    if optim is not None:
        state["optim"] = optim.state_dict()

    if scheduler is not None:
        state["scheduler"] = scheduler.state_dict()

    state["other_states"] = other_states

    torch.save(state, ckpt_path)


def load_checkpoint(
    ckpt_path: str,
    model: Optional[TModel] = None,
    optim: Optional[Optimizer] = None,
    scheduler: Optional[_LRScheduler] = None,
    set_rng_state: bool = True,
    return_other_states: bool = False,
    **torch_load_args,
) -> int: 
    
    ckpt = torch.load(ckpt_path, **torch_load_args)

    if model is not None and "net" in ckpt:
        if isinstance(model, nn.DataParallel):
            model = model.module

        model.load_state_dict(ckpt["net"])

    if optim and "optimizer" in ckpt:
        optim.load_state_dict(ckpt["optimizer"])

    if scheduler and "scheduler" in ckpt:
        scheduler.load_state_dict(ckpt["scheduler"])

    if set_rng_state and "rng_state" in ckpt:
        torch.set_rng_state(ckpt["rng_state"][0])
        np.random.set_state(ckpt["rng_state"][1])
        random.setstate(ckpt["rng_state"][2])

    if return_other_states:
        ret = (ckpt["epoch"], ckpt.get("other_states", {}))

    else:
        ret = ckpt["epoch"]

    return ret
