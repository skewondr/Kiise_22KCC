import os

import numpy as np
import torch

from torch.nn import Module, Embedding, LSTM, Linear, Dropout
from IPython import embed 

class EMB(Module):
    def __init__(self, num_c, emb_size, dropout=0.1, emb_type="qid"):
        super().__init__()
        self.model_name = "emb"
        self.num_c = num_c
        self.emb_size = emb_size
        self.input_emb = Embedding(self.num_c, self.emb_size)
        self.dropout_layer = Dropout(dropout)
        self.out_layer = Linear(self.emb_size, 1)
        self.emb_type = emb_type
        

    def forward(self, q):
        xemb = self.input_emb(q)
        h = self.dropout_layer(xemb)
        y = self.out_layer(h)

        return y.squeeze()