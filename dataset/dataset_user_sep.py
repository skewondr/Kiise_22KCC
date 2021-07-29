import torch
from torch.utils.data import Dataset
from config import ARGS
from constant import *


class UserSepDataset(Dataset):

    def __init__(self, name, sample_infos, dataset_name='ASSISTments2009'):
        self._name = name # train, val, test
        self._sample_infos = sample_infos # {"data":sample_data, "num_of_interactions":num_interacts}
        self._dataset_name = dataset_name

    def __repr__(self):
        return f'{self._name}: # of samples: {len(self._sample_infos["num_of_interactions"])}'

    def __len__(self):
        return len(self._sample_infos["num_of_interactions"])

    def __getitem__(self, index):
        return {
            "data": self._sample_infos["data"][index],
            "num_of_interactions": self._sample_infos["num_of_interactions"][index]
            }
       
    def get_sequence(self, batch):
        batch_data = [b['data'] for b in batch]
        batch_num_interacts = [b['num_of_interactions'] for b in batch]
        
        labels = []
        input_lists = []
        target_ids = []
        for data, num_of_interactions in zip(batch_data, batch_num_interacts):
            
            if num_of_interactions == ARGS.seq_size + 1:
                pad_counts = 0
            else:
                pad_counts = ARGS.seq_size + 1 - num_of_interactions

            input_list = []
            correct_list = []
            for idx, line in enumerate(data):
                line = line.rstrip().split(',')
                tag_id = int(line[0])
                is_correct = int(line[1])

                if idx == len(data) - 1:
                    target_id = tag_id
                
                if is_correct:
                    input_list.append(tag_id)
                else:
                    input_list.append(tag_id + QUESTION_NUM[self._dataset_name])
            
            correct_list.append(is_correct)

            paddings = [PAD_INDEX] * pad_counts
            input_list = paddings + input_list
            correct_list = paddings + correct_list 
            assert len(input_list) == ARGS.seq_size + 1, "sequence size error"

            labels.append([correct_list[-1]])
            input_lists.append(input_list[:-1])
            target_ids.append([target_id])
        
        return {
            'label': torch.as_tensor(labels),
            'input': torch.as_tensor(input_lists),
            'target_id': torch.as_tensor(target_ids)
        }
