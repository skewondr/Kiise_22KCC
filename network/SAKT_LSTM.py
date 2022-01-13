"""
Based on Annotated Transformer from Harvard NLP:
https://nlp.seas.harvard.edu/2018/04/03/attention.html#applications-of-attention-in-our-model
"""
import torch
import torch.nn as nn
import torch.nn.functional as F
import math
from constant import *
from config import ARGS
from network.util_network import get_pad_mask, get_subsequent_mask, clones
from network.SAKT import *
from logzero import logger

class MultiHeadedAttention_cat(MultiHeadedAttention):
    def __init__(self, h, qd, kd, vd, d_model, dropout=0.1):
        "Take in model size and number of heads."
        # super().__init__()
        super().__init__(h, d_model, d_model, d_model, dropout)
        
        # We assume d_v always equals d_k
        self.d_k = d_model
        self.d_v = d_model
        self.n_head = h

        self.w_qs = nn.Linear(qd, h * self.d_k, bias=False) # Q
        self.w_ks = nn.Linear(kd, h * self.d_k, bias=False) # K
        self.w_vs = nn.Linear(vd, h * self.d_v, bias=False) # V
        self.linear = nn.Linear(h * self.d_v, d_model, bias=False) # last
        self.linear_q = nn.Linear(qd, d_model, bias=False) 
        
        self.attention = ScaledDotProductAttention(dropout)
        self.dropout = nn.Dropout(dropout)
        self.layer_norm = nn.LayerNorm(d_model, eps=1e-6) 
    
    def forward(self, query, key, value, mask=None):
        "Implements Figure 2"
        if mask is not None:
            # Same mask applied to all h heads.
            mask = mask.unsqueeze(1)
        nbatches = query.size(0)

        # 1) Do all the linear projections in batch from d_model => h x d_k
        residual = self.linear_q(query) ## can be erased  (1024, 100, qd) -> (1024 100 dim)
        q = self.w_qs(query).view(nbatches, -1, self.n_head, self.d_k).transpose(1, 2)
        k = self.w_ks(key).view(nbatches, -1, self.n_head, self.d_k).transpose(1, 2)
        v = self.w_vs(value).view(nbatches, -1, self.n_head, self.d_v).transpose(1, 2)
        
        # 2) Apply attention on all the projected vectors in batch.
        x, self.attn =  self.attention(q, k, v, mask=mask)

        # 3) "Concat" using a view and apply a final linear.
        x = x.transpose(1, 2).contiguous() \
            .view(nbatches, -1, self.n_head * self.d_k)

        output = self.dropout(self.linear(x))  ## dropout can be erased
        output += residual  ## can be erased (1024, 100, 200)
        output = self.layer_norm(output) ## can be erased

        return output

class SAKTLayer_cat(SAKTLayer):
    """
    Single Encoder block of SAKT
    """
    def __init__(self, qd, kd, vd, hidden_dim, num_head, dropout):
        # super().__init__()
        super().__init__(hidden_dim, num_head, dropout)

        self._self_attn = MultiHeadedAttention_cat(num_head, qd, kd, vd, hidden_dim, dropout) #hidden_dim = d_model, dk, dv
        self._ffn = PositionwiseFeedForward(hidden_dim, hidden_dim, dropout) #hidden_dim = d_model, d_ff
        self._layernorms = clones(nn.LayerNorm(hidden_dim, eps=1e-6), 2)

class SAKT_LSTM(nn.Module):
    """
    <concat version>
    Transformer-based
    all hidden dimensions (d_k, d_v, ...) are the same as hidden_dim
    """
    def __init__(self, qd, cd, pd, hidden_dim, question_num, num_layers, num_head, dropout):
        super().__init__()
        self._hidden_dim = hidden_dim
        self._question_num = question_num
        self._num_layer = num_layers

        self.qd = qd+cd+pd
        self.kd = qd+cd+pd
        self.vd = qd+cd+pd
        
        self.cd = cd
        self.qd_ = qd
        self.pd = pd
        self.cosim = nn.CosineSimilarity(dim=-1, eps=1e-6)

        # Blocks
        self._first_layer = SAKTLayer_cat(self.qd, self.kd, self.vd, hidden_dim, num_head, dropout)
        self._further_layers = clones(SAKTLayer(hidden_dim, num_head, dropout), num_layers-1)

        # prediction layer
        self._prediction = nn.Linear(hidden_dim, 1)

        # Embedding layers
        self._positional_embedding = nn.Embedding(ARGS.seq_size+2, pd, padding_idx=PAD_INDEX, sparse=True)
        
        self._correctness_embedding = nn.Embedding(3+1, cd, padding_idx=PAD_INDEX, sparse=True)
        self._question_embedding = nn.Embedding(question_num+1, qd, padding_idx=PAD_INDEX, sparse=True)
       
        self._lstm = nn.LSTM(hidden_dim, hidden_dim, num_layers=num_layers, batch_first=True, dropout=dropout)

    def forward(self, X):
        """
        Query: Question (skill, exercise, ...) embedding
        Key, Value: Interaction embedding + positional embedding
        """
        interaction_vector = self._question_embedding(X['input']) #Q
        position_vector = self._positional_embedding(X['position']) 
        correctness_vector = self._correctness_embedding(X['correctness']) #3
    
        q_crt_vector = self._correctness_embedding(torch.LongTensor([UNKNOWN]).to(ARGS.device))
        q_crt_vector = torch.unsqueeze(q_crt_vector, dim=0).expand(interaction_vector.shape[0], interaction_vector.shape[1], q_crt_vector.shape[-1])

        q_pos_vector = self._positional_embedding(X['position'][:,-1]).to(ARGS.device) #batch, pos_dim
        q_pos_vector = torch.unsqueeze(q_pos_vector, dim=1).repeat(1, interaction_vector.shape[1], 1) #batch, len, pos_dim 

        q_vector = torch.cat((interaction_vector, q_crt_vector, q_pos_vector), -1)
        vk_vector = torch.cat((interaction_vector, correctness_vector, position_vector), -1) 

        input_q = q_vector[:,1:,:]
        input_vk = vk_vector[:,:-1,:]            

        mask = get_pad_mask(X['input'][:,:-1], PAD_INDEX) & get_subsequent_mask(X['input'][:,:-1])
        #mask = None 
       
        for i in range(self._num_layer):
            if i==0:
                x = self._first_layer(query=input_q, key=input_vk, value=input_vk, mask=mask)
            else:
                x = self._further_layers[i-1](query=x, key=x, value=x, mask=mask)

        x, _ = self._lstm(x)
        x = x[:,-1,:]
        
        output = self._prediction(x)
        return output 