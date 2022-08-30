import os

import numpy as np
import torch

from torch.nn import Module, Embedding, LSTM, Linear, Dropout
from IPython import embed 

class DKT(Module):
    def __init__(self, device, num_c, emb_size, dropout=0.1, emb_type='qid', emb_path="", pretrain_dim=768):
        super().__init__()
        self.model_name = "dkt"
        self.num_c = num_c
        self.emb_size = emb_size
        self.hidden_size = emb_size
        self.emb_type = emb_type
        self.emb_path = emb_path
        self.fix_dim = 512

        if emb_type == "qid":
            self.interaction_emb = Embedding(self.num_c * 2, self.emb_size)

        elif emb_type.startswith("qid_"):
            self.interaction_emb = Embedding(self.num_c, self.fix_dim)
            self.emb_layer = Linear(self.fix_dim, self.emb_size) #
            self.emb_layer2 = Linear(self.emb_size*2, self.emb_size) #
            self.emb_predict = Linear(self.emb_size, 1)
            self.drop = Dropout(0.2)

        elif emb_type == "Q_pretrain":
            net = torch.load(emb_path)
            self.interaction_emb = Embedding.from_pretrained(net["input_emb.weight"]) #
            self.emb_layer = Linear(self.fix_dim, self.emb_size) #
            self.emb_layer2 = Linear(self.emb_size*2, self.emb_size) #

        elif emb_type.startswith("R_"):
            self.token_num = int(emb_type.split("_")[-1])
            self.interaction_emb = Embedding(self.num_c, self.fix_dim)
            self.emb_layer = Linear(self.fix_dim, self.emb_size) #
            if emb_type.startswith("R_dadd"):
                self.diff_emb = Embedding(self.token_num, self.emb_size)
                self.r_emb = Embedding(2+1, self.emb_size) #
            elif emb_type.startswith("R_sinu"): 
                self.gap = int(emb_type.split("_")[-2])
                print("gap:", self.gap)
                diff_vec = torch.from_numpy(self.get_sinusoid_encoding_table(self.token_num*2, self.emb_size)).to(device)
                self.diff_emb = Embedding.from_pretrained(diff_vec, freeze=False)
                self.emb_layer2 = Linear(self.emb_size*2, self.emb_size) #
            else:
                self.diff_emb = Embedding(self.token_num*2, self.emb_size)
                self.emb_layer2 = Linear(self.emb_size*2, self.emb_size) #
        
        self.question_emb = Embedding(self.num_c, self.emb_size)
        self.lstm_layer = LSTM(self.emb_size, self.hidden_size, batch_first=True)
        self.dropout_layer = Dropout(dropout)
        self.out_layer = Linear(self.hidden_size, 1)
        self.qa_embed = Embedding(2, self.emb_size)

    def get_sinusoid_encoding_table(self, n_seq, d_hidn):
        def cal_angle(position, i_hidn):
            return position / np.power(10000, 2 * (i_hidn // 2) / d_hidn)
        def get_posi_angle_vec(position):
            return [cal_angle(position, i_hidn) for i_hidn in range(d_hidn)]
        ran = np.arange(n_seq)
        ran[self.token_num:] = ran[self.token_num:] + self.gap
        sinusoid_table = np.array([get_posi_angle_vec(i_seq) for i_seq in ran])
        sinusoid_table[:, 0::2] = np.sin(sinusoid_table[:, 0::2])  # even index sin 
        sinusoid_table[:, 1::2] = np.cos(sinusoid_table[:, 1::2])  # odd index cos

        return sinusoid_table

    def forward(self, diff, q, r, cshft):
        emb_type = self.emb_type
        if emb_type == "qid":
            x = q + self.num_c * r
            xemb = self.interaction_emb(x)
            
        elif emb_type == "Q_pretrain" or emb_type.startswith("qid_"):
            xemb = self.emb_layer(self.interaction_emb(q))
            if emb_type == "qid_emb":
                return self.emb_predict(self.drop(xemb))
            # z = torch.zeros_like(xemb)
            # xemb_o = torch.cat([z, xemb], dim=-1)
            # xemb_x = torch.cat([xemb, z], dim=-1)
            # xemb = torch.where(r.unsqueeze(-1).repeat(1, 1, self.emb_size*2) == 1 , xemb_o, xemb_x)
            # xemb = self.emb_layer2(xemb)
            z = self.qa_embed(r)
            xemb = torch.cat([xemb, z], dim=-1)
            xemb = self.emb_layer2(xemb)

        elif emb_type.startswith("R_"):
            xemb = self.emb_layer(self.interaction_emb(q))
            if emb_type.startswith("R_dadd"):
                demb = self.diff_emb(diff)
                remb = self.r_emb(r)
                xemb = xemb + remb + demb
            else:
                diff_x = diff + self.token_num
                diff_ox = torch.where(r == 1 , diff.long(), diff_x.long()) # [batch, length]
                remb = self.diff_emb(diff_ox).float()
                if emb_type.startswith("R_add") or self.emb_type.startswith("R_sinu_a"):
                    xemb = xemb + remb
                else:  
                    xemb = torch.cat([xemb, remb], dim=-1)
                    xemb = self.emb_layer2(xemb)
        
        h, _ = self.lstm_layer(xemb)
        h = self.dropout_layer(h)
        if emb_type == "qid":
            h = h+self.question_emb(cshft) #
        else:
            h = h+self.emb_layer(self.question_emb(cshft)) #
        y = self.out_layer(h)
        y = torch.sigmoid(y)

        return y.squeeze(-1) #
        # y_idx = cshft.unsqueeze(-1).type(torch.int64)-1
        # y_idx = torch.where(y_idx==-1, 0, y_idx)
        # y = torch.gather(y, -1, y_idx).squeeze()

        # return y