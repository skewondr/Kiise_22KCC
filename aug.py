from config import ARGS
from constant import *

import numpy as np
import pickle 
from logzero import logger

import re
from random import sample
import random 

acc_name = f"../dataset/{ARGS.dataset_name}/processed/1/sub{ARGS.sub_size}/train_{ARGS.sub_size}acc.pickle"
with open(acc_name, 'rb') as f: 
    acc_dict = pickle.load(f)

"""
제거할 대상이 없는 경우 : 원본 데이터를 사용하도록 

pad_counts: 원본 데이터의 zero padding length 
input_list: 원본 데이터의 input sequence (문제 정보) 
correct_list: 원본 데이터의 input sequence (정답 정보)  
tag_list : 원본 데이터의 input sequence (문제 정보)
target_crt: 마지막 문제 라벨 
crt_idx: correct_list에서 맞힌 문제 index 
incrt_idx: correct_list에서 틀린 문제 index 
lists: 저장할 대상 dict
"""

selected_n = int(ARGS.aug_prob * ARGS.seq_size)
# logger.info(acc_dict.keys())
def Del(kwargs):
    min_selected_n = min(selected_n, len(kwargs["incrt_idx"]))
    if len(kwargs["input_list"][:-1])-min_selected_n >= 5 and min_selected_n > 0:
        ################################################################################
        if ARGS.select_type == 'lp':
            prob = np.arange(1, len(kwargs["incrt_idx"])+1)[::-1]
            idx_list = np.random.choice(kwargs["incrt_idx"], min_selected_n, p=prob/sum(prob), replace=False)

        elif ARGS.select_type == 'gcr':
            prob = []
            for i in kwargs["incrt_idx"]:
                if kwargs["tag_list"][i] in acc_dict:
                    prob.append(1-acc_dict[kwargs["tag_list"][i]])
                else:
                    prob.append(1.0)
            prob = np.array(prob)

            idx_list = np.random.choice(kwargs["incrt_idx"], min_selected_n, p=prob/sum(prob), replace=False)
        else:
            idx_list = np.random.choice(kwargs["incrt_idx"], min_selected_n, replace=False)
        ################################################################################
        input_list = [v for i, v in enumerate(kwargs["input_list"]) if i not in idx_list]
        correct_list = [v for i, v in enumerate(kwargs["correct_list"]) if i not in idx_list]
        tag_list = [v for i, v in enumerate(kwargs["tag_list"]) if i not in idx_list]

        return input_list, correct_list, tag_list
    else: 
        return kwargs["input_list"], kwargs["correct_list"], kwargs["tag_list"]

def Shuf(kwargs):
    min_selected_n = min(selected_n, len(kwargs["incrt_idx"]))
    if min_selected_n > 1:
        ################################################################################
        if ARGS.select_type == 'lp':
            prob = np.arange(1, len(kwargs["incrt_idx"])+1)[::-1]
            idx_list = np.random.choice(kwargs["incrt_idx"], min_selected_n, p=prob/sum(prob), replace=False)
        elif ARGS.select_type == 'gcr':
            prob = []
            for i in kwargs["incrt_idx"]:
                if kwargs["tag_list"][i] in acc_dict:
                    prob.append(1-acc_dict[kwargs["tag_list"][i]])
                else:
                    prob.append(1.0)
            prob = np.array(prob)
            idx_list = np.random.choice(kwargs["incrt_idx"], min_selected_n, p=prob/sum(prob), replace=False)
        else:
            idx_list = np.random.choice(kwargs["incrt_idx"], min_selected_n, replace=False)
        ################################################################################
        idx_list = np.sort(idx_list)
        shuff_idx_list = idx_list.copy()
        while (idx_list == shuff_idx_list).all():
            random.shuffle(shuff_idx_list)
        

        input_list = np.array(kwargs["input_list"])
        correct_list = np.array(kwargs["correct_list"])
        tag_list = np.array(kwargs["tag_list"])

        input_list[[idx_list]] = input_list[[shuff_idx_list]]
        correct_list[[idx_list]] = correct_list[[shuff_idx_list]]
        tag_list[[idx_list]] = tag_list[[shuff_idx_list]]

        input_list = list(input_list)
        correct_list = list(correct_list)
        tag_list = list(tag_list)
       
        return input_list, correct_list, tag_list
    else: 
        return kwargs["input_list"], kwargs["correct_list"], kwargs["tag_list"]

def Swap(kwargs):
    min_selected_n = min(selected_n, len(kwargs["incrt_idx"]))
    if min_selected_n > 0:
        incrt_idx = np.array(kwargs["incrt_idx"])
        crt_idx = np.array(kwargs["crt_idx"])
        count = 0 
        while count < min_selected_n:
            try:
                x_index = int(np.random.choice(incrt_idx, 1, replace=False))
                o_index = int(np.random.choice(crt_idx[crt_idx<x_index], 1, replace=False))
            except: 
                count +=1
                continue
            kwargs["input_list"][x_index], kwargs["input_list"][o_index] = kwargs["input_list"][o_index], kwargs["input_list"][x_index]
            kwargs["correct_list"][x_index], kwargs["correct_list"][o_index] = kwargs["correct_list"][o_index], kwargs["correct_list"][x_index]
            kwargs["tag_list"][x_index], kwargs["tag_list"][o_index] = kwargs["tag_list"][o_index], kwargs["tag_list"][x_index]

            crt_idx = crt_idx[crt_idx!=o_index]
            incrt_idx = incrt_idx[incrt_idx!=x_index]
            crt_idx = np.append(crt_idx, x_index)
            incrt_idx = np.append(incrt_idx, o_index)

            count +=1
    return kwargs["input_list"], kwargs["correct_list"], kwargs["tag_list"]
 