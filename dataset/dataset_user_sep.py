import torch
import torch.nn.functional as F
from torch.utils.data import Dataset
from config import ARGS
from constant import *
import numpy as np


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
    batch_data_path = [b[0] for b in batch]
    batch_num_interacts = [b[1] for b in batch]
    
    labels = []
    input_lists = []
    target_ids = []
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

        labels.append([correct_list[-1]])
        input_lists.append(input_list[:-1])
        target_ids.append([target_id])

    return {
        'label': torch.as_tensor(labels), #(batch, 1)
        'input': torch.as_tensor(input_lists), #(batch, seq_size)
        'target_id': torch.as_tensor(target_ids)
    }

def get_sequence_fm(batch):
    batch_data_path = [b[0] for b in batch]
    batch_num_interacts = [b[1] for b in batch]
    
    labels = []
    input_lists = []
    target_ids = []
    for data_path, num_of_interactions in zip(batch_data_path, batch_num_interacts):
        with open(data_path, 'r') as f:
            data = f.readlines()
            data = data[1:] # header exists
            sliced_data = data[:num_of_interactions+1]
            user_data_length = len(sliced_data)

        if user_data_length > ARGS.seq_size + 1:
            sliced_data = sliced_data[-(ARGS.seq_size + 1):]
            user_data_length = ARGS.seq_size + 1

        arr = []
        for line in sliced_data:
            line = line.rstrip().split(',')
            item = [int(line[0]), int(line[1])]
            arr.append(item)
        sliced_array = np.array(arr)

        sliced_array = sliced_array[sliced_array[:, 0].argsort()]
        unique, counts = np.unique(sliced_array[:,0], return_counts=True)
        index = np.array((0, *np.cumsum(counts)))

        wins = []
        fails = []
        for j in range(len(index)-1):
            win, fail = 0, 0 
            for idx, i in enumerate(range(index[j], index[j+1])):
                if idx != 0: 
                    if sliced_array[i-1, 1] == 1:
                        win +=1
                    else:
                        fail+=1 
                wins.append([win])
                fails.append([fail])
                labels.append([sliced_array[i, 1]])

        x = F.one_hot(torch.tensor(sliced_array[:,0]), num_classes=QUESTION_NUM[ARGS.dataset_name]+1)
        x = torch.cat(
            [
                x,
                torch.as_tensor(wins),
                torch.as_tensor(fails)
            ], dim=1) 
        input_lists.append(x)

    return {
        'label': torch.as_tensor(labels), #(batch, 1)
        'input': torch.cat(input_lists, dim=0), #(batch, feat_size)
        'target_id': torch.empty((1))
    }
