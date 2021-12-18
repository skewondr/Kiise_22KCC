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
from torch.optim import Optimizer, optimizer
from torch.optim.lr_scheduler import _LRScheduler
from random import sample
from constant import *
from logzero import logger
import statistics

TModel = Union[nn.DataParallel, nn.Module]

def get_pickles(name):
    if name is not None :
        with open(name, 'rb') as f: 
            return pickle.load(f)
    else: 
        return None 

def get_rm_tags(mode, train_tag_name, tag_name, rm_tag_name):
    if mode != 'train':
        with open(train_tag_name, 'rb') as f: 
            train_tags = pickle.load(f)
        with open(tag_name, 'rb') as f: 
            tags = pickle.load(f)
            rm_tags = list(set(train_tags) & set(tags))
        with open(rm_tag_name, 'wb') as f: 
                pickle.dump(rm_tags, f)
    return 

def read_data_files(flag, mode, path, user_path_list, num_of_users, save_name, rm_tag_name=None):
    rm_target_tags = get_pickles(rm_tag_name)
  
    output = []
    seq_length=[]
    correct_count=0
    incorrect_count=0
    for idx, user_path in enumerate(tqdm(user_path_list, total=num_of_users, ncols=100)):
        user_id = user_path.split('/')[-1]
        user_id = int(re.sub(r'[^0-9]', '', user_id))
        with open(os.path.join(path, user_path), 'r') as f:
            lines = f.readlines()
            lines = lines[1:]  # header exists
            num_of_interactions = len(lines) # sequence length 
            seq_length.append(num_of_interactions)
            for end_index in range(MIN_LENGTH, num_of_interactions):
                sliced_data = lines[:end_index+1]
                line = sliced_data[-1].rstrip().split(',')
                end_tag = int(line[0])
                is_correct = int(line[1])
                if is_correct: correct_count+=1
                else: incorrect_count+=1
                
                if flag == 'make_tag':
                    output.append(end_tag)
                if flag == 'make_sample':
                    if mode == 'train':
                        output.append((os.path.join(path, user_path), end_index))
                    elif mode != 'train' and end_tag in rm_target_tags:
                        output.append((os.path.join(path, user_path), end_index))
                        
    if flag == 'make_sample':
        logger.info(f"mean length : {statistics.mean(seq_length)}")
        logger.info(f"median length : {statistics.median(seq_length)}")
        logger.info(f"max length : {max(seq_length)}")

        logger.info(f"correct ratio : {correct_count/len(output):.2f}")
        logger.info(f"incorrect ratio : {incorrect_count/len(output):.2f}")

        logger.info(f"# of exercise : {len(rm_target_tags)}")

    with open(save_name, 'wb') as f: 
        pickle.dump(list(set(output)), f)
    
    return output

def get_data_infos(user_base_path, i, mode, sub_size):
    """
    if tag is not exist : make tag and get tag
    if rm_tag is not exist : make rm_tag and get rm_tag
    """
    data_path = f"{user_base_path}/{i}/"
    sample_data_name = data_path+f"sub{sub_size}/{mode}_{sub_size}sample.pickle"
    rm_tag_name = data_path+f"sub{ARGS.sub_size}/{mode}_rm_{ARGS.sub_size}target.pickle"
    train_tag_name = data_path+f"sub{ARGS.sub_size}/train_{ARGS.sub_size}target.pickle"
    tag_name = data_path+f"sub{sub_size}/{mode}_{sub_size}target.pickle"
    path = os.path.join(data_path, mode)

    # get list of all files
    user_path_list = os.listdir(path)
    num_of_users = len(user_path_list)
        
    if sub_size < 100:
        if mode == 'train':
            user_path_list = sample(user_path_list, int(num_of_users*(ARGS.sub_size*0.01)))
            num_of_users = len(user_path_list)
        elif mode == 'val':
            user_path_list = sample(user_path_list, int(num_of_users*(1*0.01)))
            num_of_users = len(user_path_list)
        elif mode == 'test':
            user_path_list = sample(user_path_list, int(num_of_users*(30*0.01)))
            num_of_users = len(user_path_list)

    if not os.path.isfile(tag_name):
        _ = read_data_files("make_tag", mode, path, user_path_list, num_of_users, tag_name)
    if not os.path.isfile(rm_tag_name):
        get_rm_tags(mode, train_tag_name, tag_name, rm_tag_name)
    if not os.path.isfile(sample_data_name):
        if mode == 'train' : name = train_tag_name
        else : name = rm_tag_name
        sample_infos = read_data_files("make_sample", mode, path, user_path_list, num_of_users, sample_data_name, name)
    else: 
        sample_infos = get_pickles(sample_data_name)

    # if mode == 'train' and not os.path.isfile(acc_name):
    #     get_data_acc(sample_data_name, acc_name)

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
