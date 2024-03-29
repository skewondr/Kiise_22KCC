import os

import numpy as np
import torch

from torch.nn import Module, Parameter, Embedding, Linear, Dropout
from torch.nn.init import kaiming_normal_

class DKVMN(Module):
    def __init__(self, device, num_c, dim_s, size_m, dropout=0.2, emb_type='qid', emb_path="", pretrain_dim=768):
        super().__init__()
        self.model_name = "dkvmn"
        self.num_c = num_c
        self.dim_s = dim_s
        self.size_m = size_m
        self.emb_type = emb_type
        self.fix_dim = 512
        self.emb_size = dim_s

        if emb_type == "qid":
            self.k_emb_layer = Embedding(self.num_c, self.fix_dim)
            self.v_emb_layer = Embedding(self.num_c * 2, self.fix_dim)
            self.emb_layer = Linear(self.fix_dim, self.emb_size) #
            self.emb_layer2 = Linear(self.fix_dim, self.emb_size) #

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

        elif emb_type.startswith(("R_", "L_")):
            self.token_num = int(emb_type.split("_")[-1])
            self.interaction_emb = Embedding(self.num_c, self.fix_dim)
            self.emb_layer = Linear(self.fix_dim, self.emb_size) #
            if emb_type.startswith("R_dadd"):
                self.diff_emb = Embedding(self.token_num, self.emb_size)
                self.r_emb = Embedding(2+1, self.emb_size) #
            elif emb_type.startswith(("R_sinu", "L_sinu")): 
                self.gap = int(emb_type.split("_")[-2])
                print("gap:", self.gap)
                diff_vec = torch.from_numpy(self.get_sinusoid_encoding_table(self.token_num*2, self.emb_size)).to(device)
                self.diff_emb = Embedding.from_pretrained(diff_vec, freeze=False)
                self.emb_layer2 = Linear(self.emb_size*2, self.emb_size) #
            else:
                self.diff_emb = Embedding(self.token_num*2, self.emb_size)
                self.emb_layer2 = Linear(self.emb_size*2, self.emb_size) #
        
        self.qa_embed = Embedding(2, self.emb_size)
        self.Mk = Parameter(torch.Tensor(self.size_m, self.dim_s))
        self.Mv0 = Parameter(torch.Tensor(self.size_m, self.dim_s))

        kaiming_normal_(self.Mk)
        kaiming_normal_(self.Mv0)

        self.f_layer = Linear(self.dim_s * 2, self.dim_s)
        self.dropout_layer = Dropout(dropout)
        self.p_layer = Linear(self.dim_s, 1)

        self.e_layer = Linear(self.dim_s, self.dim_s)
        self.a_layer = Linear(self.dim_s, self.dim_s)
    
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

    def forward(self, q, r, diff, qtest=False):
        emb_type = self.emb_type
        batch_size = q.shape[0]
        if emb_type == "qid":
            x = q + self.num_c * r
            k = self.emb_layer(self.k_emb_layer(q))
            v = self.emb_layer2(self.v_emb_layer(x))
        elif emb_type == "Q_pretrain" or emb_type.startswith("qid_"):
            k = self.emb_layer(self.interaction_emb(q))
            if emb_type == "qid_emb":
                return self.emb_predict(self.drop(k))
            # z = torch.zeros_like(k)
            # xemb_o = torch.cat([z, k], dim=-1)
            # xemb_x = torch.cat([k, z], dim=-1)
            # xemb = torch.where(r.unsqueeze(-1).repeat(1, 1, self.emb_size*2) == 1 , xemb_o, xemb_x)
            # v = self.emb_layer2(xemb)
            z = self.qa_embed(r)
            xemb = torch.cat([k, z], dim=-1)
            v = self.emb_layer2(xemb)

        elif emb_type.startswith(("R_", "L_")):
            k = self.emb_layer(self.interaction_emb(q))
            if emb_type.startswith("R_dadd"):
                demb = self.diff_emb(diff)
                remb = self.r_emb(r)
                v = k + remb + demb
            else:
                diff_x = diff + self.token_num
                diff_ox = torch.where(r == 1 , diff.long(), diff_x.long()) # [batch, length]
                remb = self.diff_emb(diff_ox).float()
                if emb_type.startswith(("R_add", "R_sinu_a", "L_sinu_a")):
                    v = k + remb
                else:  
                    xemb = torch.cat([k, remb], dim=-1)
                    v = self.emb_layer2(xemb)
        
        Mvt = self.Mv0.unsqueeze(0).repeat(batch_size, 1, 1)

        Mv = [Mvt]

        w = torch.softmax(torch.matmul(k, self.Mk.T), dim=-1)

        # Write Process
        e = torch.sigmoid(self.e_layer(v))
        a = torch.tanh(self.a_layer(v))

        for et, at, wt in zip(
            e.permute(1, 0, 2), a.permute(1, 0, 2), w.permute(1, 0, 2)
        ):
            Mvt = Mvt * (1 - (wt.unsqueeze(-1) * et.unsqueeze(1))) + \
                (wt.unsqueeze(-1) * at.unsqueeze(1))
            Mv.append(Mvt)

        Mv = torch.stack(Mv, dim=1)

        # Read Process
        f = torch.tanh(
            self.f_layer(
                torch.cat(
                    [
                        (w.unsqueeze(-1) * Mv[:, :-1]).sum(-2),
                        k
                    ],
                    dim=-1
                )
            )
        )
        p = self.p_layer(self.dropout_layer(f))

        p = torch.sigmoid(p)
        # print(f"p: {p.shape}")
        p = p.squeeze(-1)
        if not qtest:
            return p
        else:
            return p, f