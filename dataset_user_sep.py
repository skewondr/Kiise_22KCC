import torch
import torch.nn.functional as F
from torch.utils.data import Dataset
from config import ARGS
from constant import *
import numpy as np
from constant import QUESTION_NUM
import time 
import re

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
       
def get_sequence(batch):
    start_time = time.time()
    batch_data_path, batch_num_interacts = zip(*batch)
    
    lists = {"labels":[], "input_lists":[], "target_ids":[]}
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
        for idx, line in enumerate(sliced_data):
            line = line.rstrip().split(',')
            tag_id = int(line[0])
            is_correct = int(line[1])

            if idx == user_data_length - 1:
                target_id = tag_id
            
            if is_correct:
                input_list.append(tag_id)
            else:
                input_list.append(tag_id + QUESTION_NUM[ARGS.dataset_name])
        
            correct_list.append(is_correct)

        paddings = [PAD_INDEX] * pad_counts
        input_list = paddings + input_list
        correct_list = paddings + correct_list 
        assert len(input_list) == ARGS.seq_size + 1, "sequence size error"

        lists["labels"].append([correct_list[-1]])
        lists["input_lists"].append(input_list[:-1])
        lists["target_ids"].append([target_id])
       
    #print("data_loader:",len(labels), f"{time.time()-start_time:.6f}") --> 0.9 avrg sec
    return {
        'label': torch.as_tensor(lists["labels"]), #(batch, 1)
        'input': torch.as_tensor(lists["input_lists"]), #(batch, seq_size)
        'target_id': torch.as_tensor(lists["target_ids"])
    }

  
def get_sequence_attn(batch):
    start_time = time.time()
    batch_data_path, batch_num_interacts = zip(*batch)
    
    lists = {"labels":[], "input_lists":[], "target_ids":[], "tag_ids":[], "positions":[]}
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
        for idx, line in enumerate(sliced_data):
            line = line.rstrip().split(',')
            tag_id = int(line[0])
            is_correct = int(line[1])
            
            if is_correct:
                input_list.append(tag_id)
            else:
                input_list.append(tag_id + QUESTION_NUM[ARGS.dataset_name])
        
            correct_list.append(is_correct)
            tag_list.append(tag_id)

        paddings = [PAD_INDEX] * pad_counts
        pos_list = paddings + list(range(1, len(input_list)+1))
        input_list = paddings + input_list
        correct_list = paddings + correct_list 
        tag_list = paddings + tag_list
        assert len(input_list) == ARGS.seq_size + 1, "sequence size error"

        lists["labels"].append([correct_list[-1]])
        lists["input_lists"].append(input_list[:-1])
        lists["target_ids"].append([tag_list[-1]])
        lists["tag_ids"].append(tag_list[:-1])
        lists["positions"].append(pos_list[:-1])

    #print("data_loader:",len(labels), f"{time.time()-start_time:.6f}") --> 0.9 avrg sec
    return {
        'label': torch.as_tensor(lists["labels"]), #(batch, 1)
        'input': torch.as_tensor(lists["input_lists"]), #(batch, seq_size)
        'target_id': torch.as_tensor(lists["target_ids"]),
        'tag_id': torch.as_tensor(lists["tag_ids"]),
        'position': torch.as_tensor(lists["positions"])
    }

def get_sequence_qkv(batch):
    start_time = time.time()
    batch_data_path, batch_num_interacts = zip(*batch)

    lists = {"labels":[], "input_lists":[], "target_ids":[], "correct_lists":[], "pos_lists":[]}
    label_o_count = 0
    label_x_count = 0

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
        for idx, line in enumerate(sliced_data):
            line = line.rstrip().split(',')
            tag_id = int(line[0])
            is_correct = int(line[1])

            if idx == user_data_length - 1:
                target_crt = is_correct
                if is_correct:
                    label_o_count += 1
                else:
                    label_x_count += 1

            if is_correct:
                correct_list.append(CORRECT)
            else:
                correct_list.append(INCORRECT)
            input_list.append(tag_id)

        append_list(pad_counts, input_list, correct_list, target_crt, lists)

    return {
        'position':torch.as_tensor(lists["pos_lists"]),
        'correctness':torch.as_tensor(lists["correct_lists"]), #(batch, seq_size)
        'label': torch.as_tensor(lists["labels"]), #(batch, 1)
        'input': torch.as_tensor(lists["input_lists"]), #(batch, seq_size)
        'target_id': torch.as_tensor(lists["target_ids"]), 
    }

def append_list(pad_counts, input_list, correct_list, target_crt, lists):
    paddings = [PAD_INDEX] * pad_counts
    pos_list = paddings + list(range(1, len(input_list)+1))

    input_list = paddings + input_list
    correct_list = paddings + correct_list
    assert len(input_list) == ARGS.seq_size + 1, "sequence size error"

    lists["pos_lists"].append(pos_list)
    lists["correct_lists"].append(correct_list)
    lists["labels"].append([target_crt])
    lists["input_lists"].append(input_list)
    lists["target_ids"].append([input_list[-1]])

