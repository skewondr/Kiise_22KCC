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

def Pre_Del(input_lists, idx_list, n):
    if ARGS.aug_prob < 1: #aug_probability
        n = int(n * len(input_lists[0]))
        if len(input_lists[0])-1-n >= 5 and len(idx_list) > 0 and n > 0:
            if ARGS.aug_seg_time:
                prob = [i for i in reversed(range(1, len(idx_list)+1))]
                prob = [i/sum(prob) for i in prob]
                removed_idx_list = np.random.choice(idx_list,min(n, len(idx_list)), p=prob, replace=False)
            elif ARGS.aug_seg_acc:
                prob = []
                for i in idx_list:
                    if input_lists[0][i] in acc_dict:
                        prob.append(1-acc_dict[input_lists[0][i]])
                    else:
                        prob.append(1.0)
                prob = [i/sum(prob) for i in prob]
                removed_idx_list = np.random.choice(idx_list,min(n, len(idx_list)), p=prob, replace=False)
            else:
                removed_idx_list = np.random.choice(idx_list,min(n, len(idx_list)), replace=False)

            # logger.info(f"{n}. bf:{input_lists[0]}")

            input_lists0 = [v for i, v in enumerate(input_lists[0]) if i not in removed_idx_list]
            input_lists1 = [v for i, v in enumerate(input_lists[1]) if i not in removed_idx_list]

            # logger.info(f"{n}. af:{input_lists0}")

    
            return (input_lists0, input_lists1)
        else: 
            return None 

def Post_Del(input_lists, idx_list, n):
    SAKT_MODELS = ['SAKT', 'SAKT_LSTM']
    incrt = INCORRECT if ARGS.model in SAKT_MODELS else 0

    if input_lists[1][-1] == incrt and ARGS.aug_prob < 1: #aug_probability
        n = int(n * len(input_lists[0]))
        if len(input_lists[0])-1-n >= 5 and len(idx_list) > 0 and n > 0:
            if ARGS.aug_seg_time:
                prob = [i for i in range(1, len(idx_list)+1)]
                prob = [i/sum(prob) for i in prob]
                removed_idx_list = np.random.choice(idx_list,min(n, len(idx_list)), p=prob, replace=False)

            else:
                removed_idx_list = np.random.choice(idx_list,min(n, len(idx_list)), replace=False)

            input_lists0 = [v for i, v in enumerate(input_lists[0]) if i not in removed_idx_list]
            input_lists1 = [v for i, v in enumerate(input_lists[1]) if i not in removed_idx_list]
    
            return (input_lists0, input_lists1)
        else: 
            return None 

def Post_Del_Acc(input_lists, crt_list, incrt_list, n):
    SAKT_MODELS = ['SAKT', 'SAKT_LSTM']
    incrt = INCORRECT if ARGS.model in SAKT_MODELS else 0

    if input_lists[1][-1] == incrt and ARGS.aug_prob < 1: #aug_probability
        n = int(n * len(input_lists[0]))
        if (len(input_lists[0])-1-n >= 5 and n > 0) and (len(crt_list) > 0 or len(incrt_list) > 0):
            crt_prob = []
            for i in crt_list:
                if input_lists[0][i] in acc_dict:
                        crt_prob.append(acc_dict[input_lists[0][i]])
                else:
                    crt_prob.append(0.0)
            crt_prob = [i/sum(crt_prob) for i in crt_prob]
            incrt_prob = []
            for i in incrt_list:
                if input_lists[0][i] in acc_dict:
                        incrt_prob.append(1-acc_dict[input_lists[0][i]])
                else:
                    incrt_prob.append(1.0)
            incrt_prob = [i/sum(incrt_prob) for i in incrt_prob]

            idx_list, prob = zip(*sorted(zip(crt_list+incrt_list, crt_prob+incrt_prob)))
            prob = [i/sum(prob) for i in prob]

            removed_idx_list = np.random.choice(idx_list,min(n, len(idx_list)), p=prob, replace=False)

            input_lists0 = [v for i, v in enumerate(input_lists[0]) if i not in removed_idx_list]
            input_lists1 = [v for i, v in enumerate(input_lists[1]) if i not in removed_idx_list]
    
            return (input_lists0, input_lists1)
        else: 
            return None 

def Pre_Shuff(input_lists, idx_list, n):
    if ARGS.aug_prob < 1: #aug_probability
        n = int(n * len(input_lists[0]))
        if len(idx_list) > 1 and n > 1:
            if ARGS.aug_seg_time:
                prob = [i for i in reversed(range(1, len(idx_list)+1))]
                prob = [i/sum(prob) for i in prob]
                n_idx = np.random.choice(idx_list, min(n, len(idx_list)), p=prob, replace=False)
            elif ARGS.aug_seg_acc:
                prob = []
                for i in idx_list:
                    if input_lists[0][i] in acc_dict:
                        prob.append(1-acc_dict[input_lists[0][i]])
                    else:
                        prob.append(1.0)
                prob = [i/sum(prob) for i in prob]
                n_idx = np.random.choice(idx_list, min(n, len(idx_list)), p=prob, replace=False)
            else:
                n_idx = np.random.choice(idx_list, min(n, len(idx_list)), replace=False)
            shuffled_n_idx = n_idx.copy()
            while (n_idx == shuffled_n_idx).all():
                random.shuffle(shuffled_n_idx)
            shuff_idx = [shuffled_n_idx[list(n_idx).index(i)] if i in n_idx else i for i in range(len(input_lists[0]))]
            input_lists0 = np.array(input_lists[0])
            input_lists1 = np.array(input_lists[1])
            input_lists0 = input_lists0[shuff_idx]
            input_lists1 = input_lists1[shuff_idx]
            return (list(input_lists0), list(input_lists1))
        else: 
            return None 


def Pre_Swap(input_lists, crt_idx, incrt_idx, n):
    if ARGS.aug_prob < 1: #aug_probability
        n = int(n * len(input_lists[0]))
        if n > 0:
            crt_idx = np.array(crt_idx)
            incrt_idx = np.array(incrt_idx)
            count = 0 
            for _ in range(n):
                if len(crt_idx) > 0 and len(incrt_idx) > 0:
                    try: 
                        o_index = np.random.choice(crt_idx, 1, replace=False)
                        x_index = np.random.choice(incrt_idx[incrt_idx>o_index], 1, replace=False)
                        # logger.info(f"{_}, o_index:{o_index}, x_index:{x_index}, bf:{input_lists[0]}")

                        input_lists[0][o_index[0]], input_lists[0][x_index[0]] = input_lists[0][x_index[0]], input_lists[0][o_index[0]]
                        input_lists[1][o_index[0]], input_lists[1][x_index[0]] = input_lists[1][x_index[0]], input_lists[1][o_index[0]]
                        
                        crt_idx = crt_idx[crt_idx!=o_index]
                        incrt_idx = incrt_idx[incrt_idx!=x_index]
                        # logger.info(f"{_}, o_index:{o_index}, x_index:{x_index}, af:{input_lists[0]}")
                        count += 1
                    except:
                        continue
                else: break
            if count > 0 : 
                return input_lists
    return None 