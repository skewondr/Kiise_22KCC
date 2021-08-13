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
import numpy as np

class FeaturesLinear(nn.Module):

    def __init__(self, field_dims, output_dim=1):
        super().__init__()
        self.fc = torch.nn.Embedding(sum(field_dims), output_dim)
        self.bias = torch.nn.Parameter(torch.zeros((output_dim,)))
        self.offsets = np.array((0, *np.cumsum(field_dims)[:-1]), dtype=np.long)

    def forward(self, x):
        """
        :param x: Long tensor of size ``(batch_size, num_fields)``
        """
        x = x + x.new_tensor(self.offsets).unsqueeze(0)
        return torch.sum(self.fc(x), dim=1) + self.bias


class FeaturesEmbedding(nn.Module):

    def __init__(self, field_dims, embed_dim):
        super().__init__()
        self.embedding = torch.nn.Embedding(sum(field_dims), embed_dim)
        self.offsets = np.array((0, *np.cumsum(field_dims)[:-1]), dtype=np.long)
        torch.nn.init.xavier_uniform_(self.embedding.weight.data)

    def forward(self, x):
        """
        :param x: Long tensor of size ``(batch_size, num_fields)``
        """

        x = x + x.new_tensor(self.offsets).unsqueeze(0)
        return self.embedding(x)

class FactorizationMachine(nn.Module):

    def __init__(self, reduce_sum=True):
        super().__init__()
        self.reduce_sum = reduce_sum

    def forward(self, x):
        """
        :param x: Float tensor of size ``(batch_size, num_fields, embed_dim)``
        """
        square_of_sum = torch.sum(x, dim=1) ** 2
        sum_of_square = torch.sum(x ** 2, dim=1)
        ix = square_of_sum - sum_of_square
        if self.reduce_sum:
            ix = torch.sum(ix, dim=1, keepdim=True)
        return 0.5 * ix

class FactorizationMachineModel(torch.nn.Module):
    """
    A pytorch implementation of Factorization Machine.
    Reference:
        S Rendle, Factorization Machines, 2010.
    """

    def __init__(self, field_dims, embed_dim):
        super().__init__()
        self.embedding = FeaturesEmbedding(field_dims, embed_dim)
        self.linear = FeaturesLinear(field_dims)
        self.fm = FactorizationMachine(reduce_sum=True)

    def forward(self, x):
        """
        :param x: Long tensor of size ``(batch_size, num_fields)``
        """
        x = self.linear(x['input']) + self.fm(self.embedding(x['input']))
        #x = self.linear(x['input'])
        #x = self.fm(self.embedding(x['input']))
        return x

from .SAKT import *
from .DKT import *

class FM_alpha(torch.nn.Module):
    """
    A pytorch implementation of Factorization Machine.
    Reference:
        S Rendle, Factorization Machines, 2010.
    """

    def __init__(self, alpha_model, field_dims, fm_hidden_dim, input_dim, embed_dim, question_num, num_layers, num_head, dropout):
        super().__init__()
        self.embedding = FeaturesEmbedding(field_dims, fm_hidden_dim)
        self.linear = FeaturesLinear(field_dims)
        self.fm = FactorizationMachine(reduce_sum=True)
        if alpha_model == 'SAKT':
            self.alpha_model = SAKT(embed_dim, question_num, num_layers, num_head, dropout).to(ARGS.device)
        elif alpha_model == 'DKT':
            self.alpha_model = DKT(input_dim, embed_dim, num_layers, question_num, dropout).to(ARGS.device)

    def forward(self, x):
        """
        :param x: Long tensor of size ``(batch_size, num_fields)``
        """
        #x = self.linear(x['fm_input']) + self.fm(self.embedding(x['fm_input'])) + self.alpha_model(x)
        x = self.linear(x['fm_input']) + self.alpha_model(x)

        return x