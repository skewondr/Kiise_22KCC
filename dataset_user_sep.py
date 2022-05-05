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
import random 
import math 

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
    def __init__(self, model_name):
        self.emb_type = ARGS.emb_type.split('_')[0]
        self.token_num = int(ARGS.emb_type.split('_')[-1]) #index except unknown token

        collate_fn = {
            'DKT':self.get_sequence,
            'DKVMN':self.get_sequence,
            'SAKT':self.get_sequence,
            }
        
        self.collate_fn = collate_fn[model_name]

    def __call__(self, batch):
        return self.collate_fn(batch)

    def get_sequence(self, batch):
        """
        preprocessing for DKT, SAKT, DKVMN
        """
        if self.emb_type != "origin":
            batch_list = {
                    'DKT': {"label":[], "target_id":[], "avg_len":[], "question":[], "crtness":[], "label_crt":[]},
                    'DKVMN':{"label":[], "target_id":[], "tag_id":[], "avg_len":[], "question":[], "crtness":[], "label_crt":[]},
                    'SAKT':{"label":[], "target_id":[], "position":[], "avg_len":[], "question":[], "crtness":[], "label_crt":[]},
                    }
        else: 
            batch_list = {
                    'DKT': {"label":[], "input":[], "target_id":[], "avg_len":[], "label_crt":[]},
                    'DKVMN':{"label":[], "input":[], "target_id":[], "tag_id":[], "avg_len":[], "label_crt":[]},
                    'SAKT':{"label":[], "input":[], "target_id":[], "position":[], "avg_len":[], "label_crt":[]},
                    }

        start_time = time.time()
        batch_data_path, batch_num_interacts = zip(*batch)
        
        lists = batch_list[ARGS.model]
        for data_path, num_of_interactions in zip(batch_data_path, batch_num_interacts):
            with open(data_path, 'r') as f:
                data = f.readlines()
                data = data[1:] # header exists
                sliced_data = data[:num_of_interactions]
                user_data_length = len(sliced_data)

            if user_data_length > ARGS.seq_size:
                sliced_data = sliced_data[-(ARGS.seq_size):]
                user_data_length = len(sliced_data)

            input_list = []
            label_list = []
            tag_list = []
            crt_token_list = []

            for idx, line in enumerate(sliced_data):
                line = line.rstrip().split(',')
                tag_id = int(line[0])
                is_correct = int(line[1])
                if is_correct:
                    input_list.append(tag_id)
                else:
                    input_list.append(tag_id + QUESTION_NUM[ARGS.dataset_name])
                
                crt_token_list.append(self.get_token(tag_id, is_correct))

                label_list.append(is_correct)
                tag_list.append(tag_id)

            self.append_list(input_list=input_list, label_list=label_list, tag_list=tag_list, crt_list=crt_token_list, lists=lists)
        #print("data_loader:",len(labels), f"{time.time()-start_time:.6f}") --> 0.9 avrg sec
        aug_batch = dict()
        for d in lists:
            # logger.info(d, lists[d][0])
            aug_batch[d] = torch.as_tensor(lists[d])
        return aug_batch

    def get_token(self, tag_id, is_correct): 
        if self.token_num > 3:
            if tag_id in ACC_DICT:
                acc = ACC_DICT[tag_id]
                if acc > 0.0:
                    if is_correct:
                        return math.ceil(acc*int(self.token_num/2))
                    else: 
                        return math.ceil(acc*int(self.token_num/2)+int(self.token_num/2))
                else: 
                    if is_correct:
                        return 1
                    else:
                        return 1+int(self.token_num/2)

        if self.token_num == 1:
            return 1

        if is_correct:
            return 1
        else:
            return 2
                
    def append_list(self, **kwargs): 
        """
        input_list: 원본 데이터의 input sequence (문제 + 정답 정보) 
        label_list: 원본 데이터의 input sequence (정답 정보)  
        tag_list : 원본 데이터의 input sequence (문제 정보)
        crt_list : 원본 데이터의 input sequence (정답 정보를 임베딩 유형에 따라 토큰으로 변환한 것)  
        lists: 저장할 대상 dict
        """
        input_list, label_list, tag_list, crt_list = kwargs["input_list"], kwargs["label_list"], kwargs["tag_list"], kwargs["crt_list"]
        pad_counts = ARGS.seq_size - len(input_list)

        paddings = [PAD_INDEX] * pad_counts
        pos_list = paddings + list(range(1, len(input_list)+1))
        
        input_len = len(input_list)
        input_list = paddings + input_list
        label_list = paddings + label_list 
        tag_list = paddings + tag_list

        assert len(input_list) == ARGS.seq_size, "sequence size error"

        kwargs["lists"]["label"].append([label_list[-1]])
        kwargs["lists"]["target_id"].append([tag_list[-1]])
        kwargs["lists"]["avg_len"].append([input_len])
        kwargs["lists"]["label_crt"].append([crt_list[-1]])
       
        if ARGS.model in ['DKVMN']:
            kwargs["lists"]["tag_id"].append(tag_list[:-1])

        elif ARGS.model in ['SAKT']:
            kwargs["lists"]["position"].append(pos_list[:-1]) 

        if self.emb_type != "origin":
            crt_list = paddings + crt_list
            kwargs["lists"]["question"].append(tag_list[:-1])
            kwargs["lists"]["crtness"].append(crt_list[:-1])
        else: 
            kwargs["lists"]["input"].append(input_list[:-1])



    