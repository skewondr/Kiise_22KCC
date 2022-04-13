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

class UserSepDataset(Dataset):

    def __init__(self, name, sample_infos, dataset_name='ASSISTments2009'):
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
            'DKVMN':self.get_sequence_attn,
            'SAKT':self.get_sequence_attn,
            }
        self.collate_fn = collate_fn[model_name]

    def __call__(self, batch):
        return self.collate_fn(batch)

    def get_sequence(self, batch):
        start_time = time.time()
        batch_data_path, batch_num_interacts = zip(*batch)
        
        lists = {"labels":[], "input_lists":[], "target_ids":[], "avg_len":[]}
        for data_path, num_of_interactions in zip(batch_data_path, batch_num_interacts):
            with open(data_path, 'r') as f:
                data = f.readlines()
                data = data[1:] # header exists
                sliced_data = data[:num_of_interactions+1]
                user_data_length = len(sliced_data)

            if user_data_length > ARGS.seq_size + 1:
                sliced_data = sliced_data[-(ARGS.seq_size + 1):]
                user_data_length = ARGS.seq_size + 1
                pad_counts = 0   
            else:
                pad_counts = ARGS.seq_size + 1 - user_data_length

            input_list = []
            correct_list = []
            crt_idx = []
            incrt_idx = []

            for idx, line in enumerate(sliced_data):
                line = line.rstrip().split(',')
                tag_id = int(line[0])
                is_correct = int(line[1])

                if idx == user_data_length - 1:
                    target_id = tag_id
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

            self.append_list(pad_counts, input_list, correct_list, target_crt, crt_idx, incrt_idx, lists, target_id)
        
        #print("data_loader:",len(labels), f"{time.time()-start_time:.6f}") --> 0.9 avrg sec
        return {
            'label': torch.as_tensor(lists["labels"]), #(batch, 1)
            'input': torch.as_tensor(lists["input_lists"]), #(batch, seq_size)
            'target_id': torch.as_tensor(lists["target_ids"]),
            'avg_len': torch.as_tensor(lists["avg_len"]),
        }

    
    def get_sequence_attn(self, batch):
        start_time = time.time()
        batch_data_path, batch_num_interacts = zip(*batch)
        
        lists = {"labels":[], "input_lists":[], "target_ids":[], "tag_ids":[], "positions":[], "avg_len":[]}
        for data_path, num_of_interactions in zip(batch_data_path, batch_num_interacts):
            with open(data_path, 'r') as f:
                data = f.readlines()
                data = data[1:] # header exists
                sliced_data = data[:num_of_interactions+1]
                user_data_length = len(sliced_data)

            if user_data_length > ARGS.seq_size + 1:
                sliced_data = sliced_data[-(ARGS.seq_size + 1):]
                user_data_length = ARGS.seq_size + 1
                pad_counts = 0   
            else:
                pad_counts = ARGS.seq_size + 1 - user_data_length

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

            self.append_list(pad_counts, input_list, correct_list, target_crt, crt_idx, incrt_idx, lists, tag_list)

        #print("data_loader:",len(labels), f"{time.time()-start_time:.6f}") --> 0.9 avrg sec
        return {
            'label': torch.as_tensor(lists["labels"]), #(batch, 1)
            'input': torch.as_tensor(lists["input_lists"]), #(batch, seq_size)
            'target_id': torch.as_tensor(lists["target_ids"]),
            'tag_id': torch.as_tensor(lists["tag_ids"]),
            'position': torch.as_tensor(lists["positions"]),
            'avg_len': torch.as_tensor(lists["avg_len"]),
        }

    def get_sequence_qkv(self, batch):
        start_time = time.time()
        batch_data_path, batch_num_interacts = zip(*batch)

        lists = {"labels":[], "input_lists":[], "target_ids":[], "correct_lists":[], "pos_lists":[], "avg_len":[]}

        for data_path, num_of_interactions in zip(batch_data_path, batch_num_interacts):
            with open(data_path, 'r') as f:
                data = f.readlines()
                data = data[1:] # header exists
                sliced_data = data[:int(num_of_interactions)+1]
                user_data_length = len(sliced_data)

            if user_data_length > ARGS.seq_size + 1:
                sliced_data = sliced_data[-(ARGS.seq_size + 1):]
                user_data_length = ARGS.seq_size + 1
                pad_counts = 0
            else:
                pad_counts = ARGS.seq_size + 1 - user_data_length

            input_list = []
            correct_list = []
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
                    correct_list.append(CORRECT)
                else:
                    if idx != user_data_length - 1:
                        incrt_idx.append(idx)
                    correct_list.append(INCORRECT)
                    
                input_list.append(tag_id)

            self.append_list(pad_counts, input_list, correct_list, target_crt, crt_idx, incrt_idx, lists)

        return {
            'position':torch.as_tensor(lists["pos_lists"]),
            'correctness':torch.as_tensor(lists["correct_lists"]), #(batch, seq_size)
            'label': torch.as_tensor(lists["labels"]), #(batch, 1)
            'input': torch.as_tensor(lists["input_lists"]), #(batch, seq_size)
            'target_id': torch.as_tensor(lists["target_ids"]), 
            'avg_len': torch.as_tensor(lists["avg_len"]),
        }

    def append_list(self, pad_counts, input_list, correct_list, target_crt, crt_idx, incrt_idx, lists, other_list=None):
        """
        pad_counts: 원본 데이터의 zero padding length 
        input_list: 원본 데이터의 input sequence (문제 정보) 
        correct_list: 원본 데이터의 input sequence (정답 정보)  
        target_crt: 마지막 문제 라벨 
        crt_idx: correct_list에서 맞힌 문제 index 
        incrt_idx: correct_list에서 틀린 문제 index 
        lists: 저장할 대상 dict
        other_list: 각 함수마다 특별히 필요한 요소들 
        """
        if self.aug_flag:
            # logger.info("go in aug_flag")

            ###################################### Deletion ############################################
            """
            제거할 대상이 없는 경우 : 원본 데이터를 사용하도록 
            """
            selected_n = int(ARGS.aug_prob * ARGS.seq_size)

            if ARGS.del_type == 'P': #label == 1
                min_selected_n = min(selected_n, len(incrt_idx))
                removed_idx_list = np.random.choice(incrt_idx, min_selected_n, replace=False)
                # logger.info("remove incrt idx")

            elif ARGS.del_type == 'N':  #label == 0
                min_selected_n = min(selected_n, len(crt_idx))
                removed_idx_list = np.random.choice(crt_idx, min_selected_n, replace=False)
                # logger.info("remove crt idx")

            elif ARGS.del_type == 'A':  #random 
                min_selected_n = min(selected_n, len(input_list[:-1]))
                removed_idx_list = np.random.choice(range(len(input_list[:-1])), min_selected_n, replace=False)
                # logger.info("remove crt idx")
            
            else : min_selected_n = 0

            if min_selected_n > 0 : 

                input_list = [v for i, v in enumerate(input_list) if i not in removed_idx_list]
                correct_list = [v for i, v in enumerate(correct_list) if i not in removed_idx_list]

                if isinstance(other_list, list):
                    other_list = [v for i, v in enumerate(other_list) if i not in removed_idx_list]
                    # logger.info("here")

                pad_counts = ARGS.seq_size + 1 - len(input_list)
            ###################################### Deletion ############################################

        paddings = [PAD_INDEX] * pad_counts
        pos_list = paddings + list(range(1, len(input_list)+1))

        input_len = len(input_list)
        input_list = paddings + input_list
        correct_list = paddings + correct_list 

        assert len(input_list) == ARGS.seq_size + 1, "sequence size error"

        if ARGS.model in ['SAKT_LSTM']: 

            lists["pos_lists"].append(pos_list)
            lists["correct_lists"].append(correct_list)
            lists["labels"].append([target_crt])
            lists["input_lists"].append(input_list)
            lists["target_ids"].append([input_list[-1]])
            lists["avg_len"].append([input_len])
        
        elif ARGS.model in ['SAKT', 'DKVMN']:

            tag_list = paddings + other_list
            assert len(tag_list) == ARGS.seq_size + 1, "sequence size error"

            lists["labels"].append([correct_list[-1]])
            lists["input_lists"].append(input_list[:-1])
            lists["target_ids"].append([tag_list[-1]])
            lists["tag_ids"].append(tag_list[:-1])
            lists["positions"].append(pos_list[:-1])
            lists["avg_len"].append([input_len])

        elif ARGS.model in ['DKT']:

            target_id = other_list

            lists["labels"].append([correct_list[-1]])
            lists["input_lists"].append(input_list[:-1])
            lists["target_ids"].append([target_id])
            lists["avg_len"].append([input_len])



    
