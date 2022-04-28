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

def get_rm_tags(mode, train_target_name, target_name, rm_target_name):
    if mode != 'train':
        with open(train_target_name, 'rb') as f: 
            train_tags = pickle.load(f)
        with open(target_name, 'rb') as f: 
            tags = pickle.load(f)
            rm_tags = list(set(train_tags) & set(tags))
        with open(rm_target_name, 'wb') as f: 
                pickle.dump(rm_tags, f)
    return 

def read_data_files(flag, mode, path, user_path_list, num_of_users, save_name, save_l_name=None, rm_target_name=None):
    rm_target_tags = get_pickles(rm_target_name)
  
    output = []
    seq_length=[]
    correct_count=0
    incorrect_count=0
    label = []
    for idx, user_path in enumerate(tqdm(user_path_list, total=num_of_users, ncols=100)):
        user_id = user_path.split('/')[-1]
        user_id = int(re.sub(r'[^0-9]', '', user_id))
        with open(os.path.join(path, user_path), 'r') as f:
            lines = f.readlines()
            lines = lines[1:]  # header exists
            num_of_interactions = len(lines) # sequence length 
            for end_index in range(MIN_LENGTH, MAX_LENGTH+1):
                sliced_data = lines[:end_index]
                seq_length.append(len(sliced_data))
                line = sliced_data[-1].rstrip().split(',')
                end_tag = int(line[0])
                is_correct = int(line[1])
                if is_correct: correct_count+=1
                else: incorrect_count+=1
                
                if flag == 'make_target':
                    output.append(end_tag)
                if flag == 'make_sample':
                    if mode == 'train':
                        output.append((os.path.join(path, user_path), end_index))
                        label.append(is_correct)
                    elif mode != 'train' and end_tag in rm_target_tags:
                        output.append((os.path.join(path, user_path), end_index))
                        label.append(is_correct)
                
    if flag == 'make_sample':
        logger.info(f"mean length : {statistics.mean(seq_length)}")
        logger.info(f"median length : {statistics.median(seq_length)}")
        logger.info(f"max length : {max(seq_length)}")

        logger.info(f"correct ratio : {correct_count/len(output):.2f}")
        logger.info(f"incorrect ratio : {incorrect_count/len(output):.2f}")

        logger.info(f"# of exercise : {len(rm_target_tags)}")

    if flag == 'make_target':
        with open(save_name, 'wb') as f: 
                pickle.dump(set(output), f)

    if flag == 'make_sample':
        with open(save_name, 'wb') as f: 
            pickle.dump(output, f)
        with open(save_l_name, 'wb') as f: 
            pickle.dump(label, f)
        
    return output, label

def get_data_infos(user_base_path, i, mode, sub_size):
    """
    if tag is not exist : make tag and get tag
    if rm_tag is not exist : make rm_tag and get rm_tag
    """
    data_path = f"{user_base_path}/{i}/"
    sample_data_name = data_path+f"sub{sub_size}/{mode}_{sub_size}sample.pickle"
    rm_target_name = data_path+f"sub{ARGS.sub_size}/{mode}_rm_{ARGS.sub_size}target.pickle"
    train_target_name = data_path+f"sub{ARGS.sub_size}/train_{ARGS.sub_size}target.pickle"
    target_name = data_path+f"sub{sub_size}/{mode}_{sub_size}target.pickle"
    acc_name = data_path+f"sub{sub_size}/{mode}_{sub_size}acc.pickle"
    label_name = data_path+f"sub{sub_size}/{mode}_{sub_size}label.pickle"
    path = os.path.join(data_path, mode)

    # get list of all files
    user_path_list = os.listdir(path)
    num_of_users = len(user_path_list)
        
    if mode == 'train':
        user_path_list = sample(user_path_list, int(num_of_users*(ARGS.sub_size*0.01)))
        num_of_users = len(user_path_list)
    elif mode == 'val':
        user_path_list = sample(user_path_list, int(num_of_users*(1*0.01)))
        num_of_users = len(user_path_list)
    elif mode == 'test':
        user_path_list = sample(user_path_list, int(num_of_users*(30*0.01)))
        num_of_users = len(user_path_list)

    if not os.path.isfile(target_name):
        """
        sub data에서 만들 수 있는 모든 타겟 문제 tag
        """
        _, _ = read_data_files("make_target", mode, path, user_path_list, num_of_users, target_name)
    if not os.path.isfile(rm_target_name):
        """
        valid, test에서 train 내 존재 하지 않는 타겟 문제 tag를 제외. 
        """
        get_rm_tags(mode, train_target_name, target_name, rm_target_name)
    if not os.path.isfile(sample_data_name) or not os.path.isfile(label_name):
        """
        train, valid, test에서 sample, label 셋 생성. 
        """
        if mode == 'train' : name = train_target_name
        else : name = rm_target_name
        sample_infos, label = read_data_files("make_sample", mode, path, user_path_list, num_of_users, sample_data_name, label_name, name)
    else: 
        sample_infos = get_pickles(sample_data_name)
        label = get_pickles(label_name)

    if mode == 'train' and not os.path.isfile(acc_name):
        """
        해당 train sub data에 대해서 문제 마다 global accuracy rate 측정. 
        """
        get_data_acc(sample_data_name, acc_name)
    
    return sample_infos, num_of_users, label

def get_data_acc(sample_data_name, save_name):
    user_path_list = get_pickles(sample_data_name)
    num_of_users = len(user_path_list)

    ex_total_cnt = {}
    ex_crt_cnt = {}
    ex_acc = {}
    for idx, (data_path, num_of_interactions) in enumerate(tqdm(user_path_list, total=num_of_users, ncols=100)):
        with open(data_path, 'r') as f:
            data = f.readlines()
            data = data[1:] # header exists
            sliced_data = data[:num_of_interactions]

        for _, line in enumerate(sliced_data):
            line = line.rstrip().split(',')
            tag_id = int(line[0])
            is_correct = int(line[1])

            if tag_id in ex_total_cnt:
                ex_total_cnt[tag_id] += 1
            else : 
                ex_total_cnt[tag_id] = 1

            if is_correct:
                if tag_id in ex_crt_cnt:
                    ex_crt_cnt[tag_id] += 1
                else : 
                    ex_crt_cnt[tag_id] = 1

    for key in ex_crt_cnt.keys():
        ex_acc[key] = ex_crt_cnt[key]/ex_total_cnt[key]

    logger.info(f"average of accuracy:{statistics.mean(list(ex_acc.values())):.2f}")

    with open(save_name, 'wb') as f: 
        pickle.dump(ex_acc, f)

    return 

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