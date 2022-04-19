import torch
import torch.nn.functional as F
from torch.utils.data import Dataset
from config import ARGS
from constant import *
import numpy as np
from constant import QUESTION_NUM
import time 
import re
from logzero import logger
from aug import *

class UserSepDataset(Dataset):

    def __init__(self, name, sample_infos, dataset_name='EdNet-KT1'):
        self._name = name # train, val, test
        self._sample_infos = sample_infos # {"data":sample_data, "num_of_interactions":num_interacts}
        self._dataset_name = dataset_name

    def __repr__(self):
        return f'{self._name}: # of samples: {len(self._sample_infos)}'

    def __len__(self):
        return len(self._sample_infos)

    def __getitem__(self, index):
        return self._sample_infos[index]
       

class MyCollator():
    def __init__(self, model_name, aug_flag=False):
        self.aug_flag = aug_flag

        collate_fn = {
            'DKT':self.get_sequence,
            'DKVMN':self.get_sequence,
            'SAKT':self.get_sequence,
            }

        self.collate_fn = collate_fn[model_name]

        self.aug_fn = {
            "deletion":Del,
            "swapping":Swap,
            "shuffling":Shuf,
            }

    def __call__(self, batch):
        return self.collate_fn(batch)

    def get_sequence(self, batch):
        """
        preprocessing for DKT, SAKT, DKVMN
        """
        batch_list = {
            'DKT': {"label":[], "input":[], "target_id":[], "avg_len":[]},
            'DKVMN':{"label":[], "input":[], "target_id":[], "tag_id":[], "position":[], "avg_len":[]},
            'SAKT':{"label":[], "input":[], "target_id":[], "tag_id":[], "position":[], "avg_len":[]},
        }
        start_time = time.time()
        batch_data_path, batch_num_interacts = zip(*batch)
        
        lists = batch_list[ARGS.model]
        for data_path, num_of_interactions in zip(batch_data_path, batch_num_interacts):
            with open(data_path, 'r') as f:
                data = f.readlines()
                data = data[1:] # header exists
                sliced_data = data[:num_of_interactions+1]
                user_data_length = len(sliced_data)

            if user_data_length > ARGS.seq_size + 1:
                sliced_data = sliced_data[-(ARGS.seq_size + 1):]
                user_data_length = ARGS.seq_size + 1

            input_list = []
            correct_list = []
            tag_list = []
            crt_idx = []
            incrt_idx = []

            for idx, line in enumerate(sliced_data):
                line = line.rstrip().split(',')
                tag_id = int(line[0])
                is_correct = int(line[1])

                if idx == user_data_length - 1:
                    target_crt = is_correct
                
                if is_correct:
                    if idx != user_data_length - 1:
                        crt_idx.append(idx)
                    input_list.append(tag_id)
                else:
                    if idx != user_data_length - 1:
                        incrt_idx.append(idx)
                    input_list.append(tag_id + QUESTION_NUM[ARGS.dataset_name])
            
                correct_list.append(is_correct)
                tag_list.append(tag_id)

            self.append_list(input_list=input_list, correct_list=correct_list, tag_list=tag_list, target_crt=target_crt, crt_idx=crt_idx, incrt_idx=incrt_idx, lists=lists)
        
        #print("data_loader:",len(labels), f"{time.time()-start_time:.6f}") --> 0.9 avrg sec
        aug_batch = dict()
        for d in lists:
            aug_batch[d] = torch.as_tensor(lists[d])
        return aug_batch
    
    def append_list(self, **kwargs): 
        """
        input_list: 원본 데이터의 input sequence (문제 + 정답 정보) 
        correct_list: 원본 데이터의 input sequence (정답 정보)  
        tag_list : 원본 데이터의 input sequence (문제 정보)
        target_crt: 마지막 문제 라벨 
        crt_idx: correct_list에서 맞힌 문제 index 
        incrt_idx: correct_list에서 틀린 문제 index 
        lists: 저장할 대상 dict
        """
        ###################################### AUGMENTATION ############################################
        if self.aug_flag:
            # logger.info("go in aug_flag")
            input_list, correct_list, tag_list = self.aug_fn[ARGS.aug_type](kwargs)
        ###################################### AUGMENTATION ############################################

        pad_counts = ARGS.seq_size + 1 - len(input_list)

        paddings = [PAD_INDEX] * pad_counts
        pos_list = paddings + list(range(1, len(input_list)+1))

        input_len = len(input_list)
        input_list = paddings + input_list
        correct_list = paddings + correct_list 
        tag_list = paddings + tag_list

        assert len(input_list) == ARGS.seq_size + 1, "sequence size error"
        
        if ARGS.model in ['DKT']:
            kwargs["lists"]["label"].append([correct_list[-1]])
            kwargs["lists"]["input"].append(input_list[:-1])
            kwargs["lists"]["target_id"].append([tag_list[-1]])
            kwargs["lists"]["avg_len"].append([input_len])

        elif ARGS.model in ['SAKT', 'DKVMN']:
            kwargs["lists"]["label"].append([correct_list[-1]])
            kwargs["lists"]["input"].append(input_list[:-1])
            kwargs["lists"]["target_id"].append([tag_list[-1]])
            kwargs["lists"]["tag_id"].append(tag_list[:-1])
            kwargs["lists"]["position"].append(pos_list[:-1])
            kwargs["lists"]["avg_len"].append([input_len])



    
