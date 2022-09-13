import torch

from torch.nn import Module, Embedding, Linear, MultiheadAttention, LayerNorm, Dropout
from .utils import transformer_FFN, pos_encode, ut_mask, get_clones
import numpy as np
from IPython import embed

class SAKT(Module):
    def __init__(self, device, num_c, seq_len, emb_size, num_attn_heads, dropout, num_en=2, emb_type="qid", emb_path="", pretrain_dim=768):
        super().__init__()
        self.device = device 
        self.model_name = "sakt"
        self.emb_type = emb_type

        self.num_c = num_c
        self.seq_len = seq_len
        self.emb_size = emb_size
        self.num_attn_heads = num_attn_heads
        self.dropout = dropout
        self.num_en = num_en
        self.fix_dim = 512

        if emb_type.startswith("qid"):
            # num_c, seq_len, emb_size, num_attn_heads, dropout, emb_path="")
            self.interaction_emb = Embedding(num_c * 2, emb_size)
            self.exercise_emb = Embedding(num_c, emb_size)
            self.emb_layer = Linear(self.fix_dim, self.emb_size) #
            self.emb_layer2 = Linear(self.fix_dim, self.emb_size) #
            # self.P = Parameter(torch.Tensor(self.seq_len, self.emb_size))

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

        self.position_emb = Embedding(seq_len, emb_size)

        self.blocks = get_clones(Blocks(device, emb_size, num_attn_heads, dropout), self.num_en)

        self.dropout_layer = Dropout(dropout)
        self.pred = Linear(self.emb_size, 1)
    
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

    def base_emb(self, diff, q, r, qry):
        if self.emb_type == "qid":
            x = q + self.num_c * r
            qshftemb, xemb = self.emb_layer(self.exercise_emb(qry)), self.emb_layer2(self.interaction_emb(x))
        
        elif self.emb_type.startswith("R_"):
            qshftemb = self.emb_layer(self.interaction_emb(qry))
            xemb = self.emb_layer(self.interaction_emb(q))
            if self.emb_type.startswith("R_dadd"):
                demb = self.diff_emb(diff)
                remb = self.r_emb(r)
                xemb = xemb + remb + demb
            else:
                diff_x = diff + self.token_num
                diff_ox = torch.where(r == 1 , diff.long(), diff_x.long()) # [batch, length]
                remb = self.diff_emb(diff_ox).float()
                if self.emb_type.startswith("R_add") or self.emb_type.startswith("R_sinu_a"):
                    xemb = xemb + remb
                else: 
                    xemb = torch.cat([xemb, remb], dim=-1)
                    xemb = self.emb_layer2(xemb)

        posemb = self.position_emb(pos_encode(self.device, xemb.shape[1]))
        xemb = xemb + posemb
        return qshftemb, xemb

    def forward(self, diff, q, r, qry, qtest=False):
        emb_type = self.emb_type
        qemb, qshftemb, xemb = None, None, None
        qshftemb, xemb = self.base_emb(diff, q, r, qry)
        # print(f"qemb: {qemb.shape}, xemb: {xemb.shape}, qshftemb: {qshftemb.shape}")
        for i in range(self.num_en):
            xemb = self.blocks[i](qshftemb, xemb, xemb)

        p = torch.sigmoid(self.pred(self.dropout_layer(xemb))).squeeze(-1)
        if not qtest:
            return p
        else:
            return p, xemb

class Blocks(Module):
    def __init__(self, device, emb_size, num_attn_heads, dropout) -> None:
        super().__init__()
        self.device = device
        self.attn = MultiheadAttention(emb_size, num_attn_heads, dropout=dropout)
        self.attn_dropout = Dropout(dropout)
        self.attn_layer_norm = LayerNorm(emb_size)

        self.FFN = transformer_FFN(emb_size, dropout)
        self.FFN_dropout = Dropout(dropout)
        self.FFN_layer_norm = LayerNorm(emb_size)

    def forward(self, q=None, k=None, v=None):
        q, k, v = q.permute(1, 0, 2), k.permute(1, 0, 2), v.permute(1, 0, 2)
        # attn -> drop -> skip -> norm 
        # transformer: attn -> drop -> skip -> norm transformer default
        causal_mask = ut_mask(self.device, seq_len = k.shape[0])
        attn_emb, _ = self.attn(q, k, v, attn_mask=causal_mask)

        attn_emb = self.attn_dropout(attn_emb)
        attn_emb, q = attn_emb.permute(1, 0, 2), q.permute(1, 0, 2)

        attn_emb = self.attn_layer_norm(q + attn_emb)

        emb = self.FFN(attn_emb)
        emb = self.FFN_dropout(emb)
        emb = self.FFN_layer_norm(attn_emb + emb)
        return emb