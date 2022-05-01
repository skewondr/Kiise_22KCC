import torch
import torch.nn as nn
from constant import PAD_INDEX
from torch.autograd import Variable
from config import ARGS
from logzero import logger

class DKT(nn.Module):
    """
    LSTM based model
    """
    def __init__(self, input_dim, hidden_dim, num_layers, question_num, dropout):
        super().__init__()
        self._hidden_dim = hidden_dim
        self._num_layers = num_layers
        if ARGS.emb_type == "origin":
            self._encoder = nn.Embedding(num_embeddings=2*question_num+1, embedding_dim=input_dim, padding_idx=PAD_INDEX, sparse=True)
            self.input_dim = input_dim 
        else: 
            self.comb_type = ARGS.emb_type.split('_')[0] #concat / add 
            self.token_num = int(ARGS.emb_type.split('_')[-1]) #index except unknown token
            if self.comb_type == 'concat':
                self._question_embedding = nn.Embedding(question_num+1, ARGS.qd, padding_idx=PAD_INDEX, sparse=True)
                self._correctness_embedding = nn.Embedding(self.token_num+1, ARGS.cd, padding_idx=PAD_INDEX, sparse=True)
                self.input_dim = ARGS.qd + ARGS.cd
            elif self.comb_type == 'add':
                self._question_embedding = nn.Embedding(question_num+1, input_dim, padding_idx=PAD_INDEX, sparse=True)
                self._correctness_embedding = nn.Embedding(self.token_num+1, input_dim, padding_idx=PAD_INDEX, sparse=True)
                self.input_dim = input_dim
        self._lstm = nn.LSTM(self.input_dim, hidden_dim, num_layers=num_layers, batch_first=True, dropout=dropout)
        self._decoder = nn.Linear(hidden_dim, question_num)

    def forward(self, x):
        """
        get model output (before taking sigmoid) for target_id
        input: (batch_size, sequence_size)
        target_id: (batch_size)
        return output, a tensor of shape (batch_size, 1)
        """
        x_input = self.make_emb(x)
        output, _ = self._lstm(x_input)
        output = self._decoder(output[:, -1, :])
        output = torch.gather(output, -1, x['target_id']-1)
        return output

    def make_emb(self, x): #kwargs : {question, crtness}
        if ARGS.emb_type == "origin":
            input_emb = self._encoder(x['input'])
        else:
            q_vector = self._question_embedding(x['question']) #Q
            c_vector = self._correctness_embedding(x['crtness']) #3
            if self.comb_type == 'concat':
                input_emb = torch.cat((q_vector, c_vector), -1) #batch, len, qd+cd
            elif self.comb_type == 'add':
                input_emb = q_vector + c_vector 
        return input_emb 