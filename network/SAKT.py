"""
Based on Annotated Transformer from Harvard NLP:
https://nlp.seas.harvard.edu/2018/04/03/attention.html#applications-of-attention-in-our-model
"""
import torch
import torch.nn as nn
import torch.nn.functional as F
import math
from constant import PAD_INDEX
from config import ARGS
from network.util_network import get_pad_mask, get_subsequent_mask, clones

class ScaledDotProductAttention(nn.Module):
    def __init__(self, attn_dropout=0.1):
        "Compute 'Scaled Dot Product Attention'"
        super().__init__()
        self.dropout = nn.Dropout(attn_dropout)

    def forward(self, query, key, value, mask=None):
        d_k = query.size(-1)
        scores = torch.matmul(query, key.transpose(-2, -1)) \
                / math.sqrt(d_k)
        if mask is not None:
            scores = scores.masked_fill(mask == 0, -1e9)
        p_attn = F.softmax(scores, dim=-1)
        p_attn = self.dropout(p_attn)
        return torch.matmul(p_attn, value), p_attn


class MultiHeadedAttention(nn.Module):
    def __init__(self, h, d_model, dk, dv, dropout=0.1):
        "Take in model size and number of heads."
        super(MultiHeadedAttention, self).__init__()
        
        assert d_model % h == 0
        # We assume d_v always equals d_k
        self.d_k = dk
        self.d_v = dv
        self.n_head = h

        self.w_qs = nn.Linear(d_model, h * dk, bias=False) # Q
        self.w_ks = nn.Linear(d_model, h * dk, bias=False) # K
        self.w_vs = nn.Linear(d_model, h * dv, bias=False) # V
        self.linear = nn.Linear(h * dv, d_model, bias=False) # last
        
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
        residual = query ## can be erased
        q = self.w_qs(query).view(nbatches, -1, self.n_head, self.d_k).transpose(1, 2)
        k = self.w_ks(key).view(nbatches, -1, self.n_head, self.d_k).transpose(1, 2)
        v = self.w_vs(value).view(nbatches, -1, self.n_head, self.d_v).transpose(1, 2)
        
        # 2) Apply attention on all the projected vectors in batch.
        x, self.attn =  self.attention(q, k, v, mask=mask)

        # 3) "Concat" using a view and apply a final linear.
        x = x.transpose(1, 2).contiguous() \
            .view(nbatches, -1, self.n_head * self.d_k)

        output = self.dropout(self.linear(x))  ## dropout can be erased
        output += residual  ## can be erased
        output = self.layer_norm(output) ## can be erased

        return output


class PositionwiseFeedForward(nn.Module):
    "Implements FFN equation."
    def __init__(self, d_model, d_ff, dropout=0.1):
        super(PositionwiseFeedForward, self).__init__()
        self.w_1 = nn.Linear(d_model, d_ff)
        self.w_2 = nn.Linear(d_ff, d_model)

        self.layer_norm = nn.LayerNorm(d_model, eps=1e-6)
        self.dropout = nn.Dropout(dropout)

    def forward(self, x):
        residual = x ## can be erased
        x = self.dropout(self.w_2(F.relu(self.w_1(x))))
        x += residual ## can be erased
        x = self.layer_norm(x) ## can be erased
        return x

class SAKTLayer(nn.Module):
    """
    Single Encoder block of SAKT
    """
    def __init__(self, hidden_dim, num_head, dropout):
        super().__init__()
        self._self_attn = MultiHeadedAttention(num_head, hidden_dim, hidden_dim, hidden_dim, dropout) #hidden_dim = d_model, dk, dv
        self._ffn = PositionwiseFeedForward(hidden_dim, hidden_dim, dropout) #hidden_dim = d_model, d_ff
        self._layernorms = clones(nn.LayerNorm(hidden_dim, eps=1e-6), 2)

    def forward(self, query, key, value, mask=None):
        """
        query: question embeddings
        key: interaction embeddings
        """
        # self-attention block
        output = self._self_attn(query=query, key=key, value=value, mask=mask)
        #output = self._layernorms[0](key + output)
        # feed-forward block
        #output = self._layernorms[1](output + self._ffn(output))
        output = self._ffn(output)
        return output


class SAKT(nn.Module):
    """
    Transformer-based
    all hidden dimensions (d_k, d_v, ...) are the same as hidden_dim
    """
    def __init__(self, hidden_dim, question_num, num_layers, num_head, dropout):
        super().__init__()
        self._hidden_dim = hidden_dim
        self._question_num = question_num

        # Blocks
        self._layers = clones(SAKTLayer(hidden_dim, num_head, dropout), num_layers)

        # prediction layer
        self._prediction = nn.Linear(hidden_dim, 1)

        # Embedding layers
        self._positional_embedding = nn.Embedding(ARGS.seq_size+1, hidden_dim, padding_idx=PAD_INDEX, sparse=True)
        self._interaction_embedding = nn.Embedding(2*question_num+1, hidden_dim, padding_idx=PAD_INDEX, sparse=True)
        self._question_embedding = nn.Embedding(question_num+1, hidden_dim, padding_idx=PAD_INDEX, sparse=True)

    def forward(self, X):
        """
        Query: Question (skill, exercise, ...) embedding
        Key, Value: Interaction embedding + positional embedding
        """
        interaction_vector = self._interaction_embedding(X['input'])
        question_vector = self._question_embedding(X['target_id'])
        position_vector = self._positional_embedding(X['position'])

        mask = get_pad_mask(X['input'], PAD_INDEX) & get_subsequent_mask(X['input'])
        mask = None 
        x = interaction_vector + position_vector

        for layer in self._layers:
            x = layer(query=question_vector, key=x, mask=mask)
        output = self._prediction(x)
        output = output[:, -1, :]
        return output
