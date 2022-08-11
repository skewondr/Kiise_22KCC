#!/usr/bin/env python
# coding=utf-8

import os, sys
import pandas as pd
import torch
from torch.utils.data import Dataset
from torch.cuda import FloatTensor, LongTensor
import numpy as np

class KTDataset(Dataset):
    """Dataset for KT
        can use to init dataset for: (for models except dkt_forget)
            train data, valid data
            common test data(concept level evaluation), real educational scenario test data(question level evaluation).

    Args:
        file_path (str): train_valid/test file path
        input_type (list[str]): the input type of the dataset, values are in ["questions", "concepts"]
        folds (set(int)): the folds used to generate dataset, -1 for test data
        qtest (bool, optional): is question evaluation or not. Defaults to False.
    """
    def __init__(self, device, file_path, data_config, folds, qtest=False):
        super(KTDataset, self).__init__()
        self.device = device
        sequence_path = file_path
        self.input_type = data_config["input_type"]
        self.q_num = data_config["num_q"]
        self.c_num = data_config["num_c"]
        self.qtest = qtest
        folds = sorted(list(folds))
        folds_str = "_" + "_".join([str(_) for _ in folds])
        if self.qtest:
            processed_data = file_path + folds_str + "_qtest.pkl"
        else:
            processed_data = file_path + folds_str + ".pkl"

        if not os.path.exists(processed_data):
            print(f"Start preprocessing {file_path} fold: {folds_str}...")
            if self.qtest:
                self.q_seqs, self.c_seqs, self.r_seqs, self.mask_seqs, self.select_masks, self.dqtest = self.__load_data__(sequence_path, folds)
                save_data = [self.q_seqs, self.c_seqs, self.r_seqs, self.mask_seqs, self.select_masks, self.dqtest]
            else:
                self.q_seqs, self.c_seqs, self.r_seqs, self.mask_seqs, self.select_masks = self.__load_data__(sequence_path, folds)
                save_data = [self.q_seqs, self.c_seqs, self.r_seqs, self.mask_seqs, self.select_masks]
            pd.to_pickle(save_data, processed_data)
        else:
            print(f"Read data from processed file: {processed_data}")
            if self.qtest:
                self.q_seqs, self.c_seqs, self.r_seqs, self.mask_seqs, self.select_masks, self.dqtest = pd.read_pickle(processed_data)
            else:
                self.q_seqs, self.c_seqs, self.r_seqs, self.mask_seqs, self.select_masks = pd.read_pickle(processed_data)
                self.q_seqs, self.c_seqs, self.r_seqs, self.mask_seqs, self.select_masks = self.q_seqs.to(self.device), self.c_seqs.to(self.device), self.r_seqs.to(self.device), self.mask_seqs.to(self.device), self.select_masks.to(self.device)
        print(f"file path: {file_path}, qlen: {len(self.q_seqs)}, clen: {len(self.c_seqs)}, rlen: {len(self.r_seqs)}")

        qc_diff_data = os.path.dirname(file_path) + "/train_valid_sequences.csv" + "_qc_diff.npz"

        if not os.path.exists(qc_diff_data):
            print(f"Start preprocessing diff {file_path} fold: {folds_str}...")
            self.q_diff, self.c_diff = self.__load_diff__(sequence_path)
            np.savez(qc_diff_data, q_diff=self.q_diff, c_diff=self.c_diff)
        else: 
            print(f"Read diff from processed file:{qc_diff_data}")
            diff_data = np.load(qc_diff_data)
            self.q_diff, self.c_diff = diff_data['q_diff'], diff_data['c_diff']
      
        q_ = self.q_diff.astype('float')
        q_[q_ == 0] = np.nan
        c_ = self.c_diff.astype('float')
        c_[c_ == 0] = np.nan
        if len(q_)>0: print(f"average of question accuracy:{np.nanmean(q_):.2f}") 
        if len(c_)>0: print(f"average of concept accuracy:{np.nanmean(c_):.2f}") 

    def __len__(self):
        """return the dataset length

        Returns:
            int: the length of the dataset
        """
        return len(self.r_seqs)

    def __getitem__(self, index):
        """
        Args:
            index (int): the index of the data want to get

        Returns:
            (tuple): tuple containing:
            
            - **q_seqs (torch.tensor)**: question id sequence of the 0~seqlen-2 interactions *
            - **c_seqs (torch.tensor)**: knowledge concept id sequence of the 0~seqlen-2 interactions 
            - **r_seqs (torch.tensor)**: response id sequence of the 0~seqlen-2 interactions
            - **qshft_seqs (torch.tensor)**: question id sequence of the 1~seqlen-1 interactions *
            - **cshft_seqs (torch.tensor)**: knowledge concept id sequence of the 1~seqlen-1 interactions
            - **rshft_seqs (torch.tensor)**: response id sequence of the 1~seqlen-1 interactions
            - **mask_seqs (torch.tensor)**: masked value sequence, shape is seqlen-1
            - **select_masks (torch.tensor)**: is select to calculate the performance or not, 0 is not selected, 1 is selected, only available for 1~seqlen-1, shape is seqlen-1
            - **dcur (dict)**: used only self.qtest is True, for question level evaluation
        """
        q_seqs, qshft_seqs, c_seqs, cshft_seqs = torch.tensor([]).to(self.device), torch.tensor([]).to(self.device), torch.tensor([]).to(self.device), torch.tensor([]).to(self.device)
        qshft_diffs, cshft_diffs = torch.tensor([]).to(self.device), torch.tensor([]).to(self.device)
        if "questions" in self.input_type:
            q_seqs = self.q_seqs[index][:-1] * self.mask_seqs[index].to(self.device)
            qshft_seqs = self.q_seqs[index][1:] * self.mask_seqs[index].to(self.device)
            _ = torch.where(qshft_seqs.long() <=0, self.q_num, qshft_seqs.long()) 
            _ = self.q_diff[_.cpu().int()]
            qshft_diffs = torch.from_numpy(_).float().to(self.device)
        if "concepts" in self.input_type:
            c_seqs = self.c_seqs[index][:-1] * self.mask_seqs[index].to(self.device)
            cshft_seqs = self.c_seqs[index][1:] * self.mask_seqs[index].to(self.device)
            _ = torch.where(cshft_seqs.long() <=0, self.c_num, cshft_seqs.long()) 
            _ = self.c_diff[_.cpu().int()]
            cshft_diffs = torch.from_numpy(_).float().to(self.device)

        r_seqs = self.r_seqs[index][:-1] * self.mask_seqs[index].to(self.device)
        rshft_seqs = self.r_seqs[index][1:] * self.mask_seqs[index].to(self.device)

        mask_seqs = self.mask_seqs[index].to(self.device)
        select_masks = self.select_masks[index].to(self.device)
        if not self.qtest:
            return q_seqs, c_seqs, r_seqs, qshft_seqs, cshft_seqs, rshft_seqs, mask_seqs, select_masks, qshft_diffs, cshft_diffs
        else:
            dcur = dict()
            for key in self.dqtest:
                dcur[key] = self.dqtest[key][index]
            return q_seqs, c_seqs, r_seqs, qshft_seqs, cshft_seqs, rshft_seqs, mask_seqs, select_masks, dcur, qshft_diffs, cshft_diffs

    def __load_data__(self, sequence_path, folds, pad_val=-1):
        """
        Args:
            sequence_path (str): file path of the sequences
            folds (list[int]): 
            pad_val (int, optional): pad value. Defaults to -1.

        Returns: 
            (tuple): tuple containing

            - **q_seqs (torch.tensor)**: question id sequence of the 0~seqlen-1 interactions
            - **c_seqs (torch.tensor)**: knowledge concept id sequence of the 0~seqlen-1 interactions
            - **r_seqs (torch.tensor)**: response id sequence of the 0~seqlen-1 interactions
            - **mask_seqs (torch.tensor)**: masked value sequence, shape is seqlen-1
            - **select_masks (torch.tensor)**: is select to calculate the performance or not, 0 is not selected, 1 is selected, only available for 1~seqlen-1, shape is seqlen-1
            - **dqtest (dict)**: not null only self.qtest is True, for question level evaluation
        """

        seq_qids, seq_cids, seq_rights, seq_mask = [], [], [], []
        df = pd.read_csv(sequence_path)
        df = df[df["fold"].isin(folds)]
        interaction_num = 0
        # seq_qidxs, seq_rests = [], []
        dqtest = {"qidxs": [], "rests":[], "orirow":[]}
        for i, row in df.iterrows():
            #use kc_id or question_id as input
            if "concepts" in self.input_type:
                seq_cids.append([int(_) for _ in row["concepts"].split(",")])
            if "questions" in self.input_type:
                seq_qids.append([int(_) for _ in row["questions"].split(",")])

            seq_rights.append([int(_) for _ in row["responses"].split(",")])
            seq_mask.append([int(_) for _ in row["selectmasks"].split(",")])

            interaction_num += seq_mask[-1].count(1)

            if self.qtest:
                dqtest["qidxs"].append([int(_) for _ in row["qidxs"].split(",")])
                dqtest["rests"].append([int(_) for _ in row["rest"].split(",")])
                dqtest["orirow"].append([int(_) for _ in row["orirow"].split(",")])

        q_seqs, c_seqs, r_seqs = FloatTensor(seq_qids).to(self.device), FloatTensor(seq_cids).to(self.device), FloatTensor(seq_rights).to(self.device)
        seq_mask = LongTensor(seq_mask).to(self.device)

        mask_seqs = (c_seqs[:,:-1] != pad_val) * (c_seqs[:,1:] != pad_val)
        select_masks = (seq_mask[:, 1:] != pad_val)#(seq_mask[:,:-1] != pad_val) * (seq_mask[:,1:] != pad_val)
        print(f"interaction_num: {interaction_num}")

        if self.qtest:
            for key in dqtest:
                dqtest[key] = LongTensor(dqtest[key])[:, 1:].to(self.device)
            return q_seqs, c_seqs, r_seqs, mask_seqs, select_masks, dqtest
        
        return q_seqs, c_seqs, r_seqs, mask_seqs, select_masks

    def __load_diff__(self, sequence_path):
        q_total_cnt = np.ones((self.q_num+1))
        q_crt_cnt = np.zeros((self.q_num+1))
        q_diff = np.zeros((self.q_num+1))

        c_total_cnt = np.ones((self.c_num+1))
        c_crt_cnt = np.zeros((self.c_num+1))
        c_diff = np.zeros((self.c_num+1))

        df = pd.read_csv(sequence_path)
        for i, row in df.iterrows():
            if "concepts" in self.input_type:
                for c, r in zip(row["concepts"].split(","), row["responses"].split(",")):
                    if int(c) == -1 : continue 
                    c_total_cnt[int(c)] += 1
                    if int(r): 
                        c_crt_cnt[int(c)] += 1
            if "questions" in self.input_type:
                for q, r in zip(row["questions"].split(","), row["responses"].split(",")):
                    if int(q) == -1 : continue 
                    q_total_cnt[int(q)] += 1
                    if int(r): 
                        q_crt_cnt[int(q)] += 1
        
        q_diff = q_crt_cnt/q_total_cnt
        c_diff = c_crt_cnt/c_total_cnt

        return q_diff, c_diff