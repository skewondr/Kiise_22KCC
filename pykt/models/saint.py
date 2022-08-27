import torch 
import torch.nn as nn
from torch.nn import Dropout
import pandas as pd
from .utils import transformer_FFN, get_clones, ut_mask, pos_encode
from torch.nn import Embedding, Linear
from IPython import embed 
import numpy as np

# device = "cpu" if not torch.cuda.is_available() else "cuda"

class SAINT(nn.Module):
    def __init__(self, device, num_q, num_c, seq_len, emb_size, num_attn_heads, dropout, n_blocks=1, emb_type="qid", emb_path=""):
        super().__init__()
        # print(f"num_q: {num_q}, num_c: {num_c}")
        if num_q == num_c and num_q == 0:
            assert num_q != 0
        self.num_q = num_q
        self.num_c = num_c
        self.model_name = "saint"
        self.num_en = n_blocks
        self.num_de = n_blocks
        self.emb_type = emb_type
        self.fix_dim = 512

        self.embd_pos = nn.Embedding(seq_len, embedding_dim = emb_size) 
        # self.embd_pos = Parameter(torch.Tensor(seq_len-1, emb_size))
        # kaiming_normal_(self.embd_pos)
        self.device = device
       
        self.encoder = get_clones(Encoder_block(device, emb_size, num_attn_heads, num_q, num_c, seq_len, dropout, emb_type, emb_path, self.fix_dim), self.num_en)
        self.decoder = get_clones(Decoder_block(device, emb_size, 2, num_attn_heads, seq_len, dropout, emb_type), self.num_de)

        self.dropout = Dropout(dropout)
        self.out = nn.Linear(in_features=emb_size, out_features=1)
    
    def forward(self, diff, in_ex, in_cat, in_res, qtest=False):
        emb_type = self.emb_type        

        if self.num_q > 0:
            in_pos = pos_encode(self.device, in_ex.shape[1])
        else:
            in_pos = pos_encode(self.device, in_cat.shape[1])
        in_pos = self.embd_pos(in_pos)
        # in_pos = self.embd_pos.unsqueeze(0)
        ## pass through each of the encoder blocks in sequence
        first_block = True
        for i in range(self.num_en):
            if i >= 1:
                first_block = False
            in_ex = self.encoder[i](in_ex, in_cat, in_pos, first_block=first_block)
            in_cat = in_ex
        ## pass through each decoder blocks in sequence
        start_token = torch.tensor([[2]]).repeat(in_res.shape[0], 1).to(self.device)
        in_res = torch.cat((start_token, in_res), dim=-1)
        r = in_res
        first_block = True
        for i in range(self.num_de):
            if i >= 1:
                first_block = False
            in_res = self.decoder[i](diff, in_res, in_pos, en_out=in_ex, first_block=first_block)
        
        ## Output layer

        res = self.out(self.dropout(in_res))
        res = torch.sigmoid(res).squeeze(-1)
        if not qtest:
            return res
        else:
            return res, in_res


class Encoder_block(nn.Module):
    """
    M = SkipConct(Multihead(LayerNorm(Qin;Kin;Vin)))
    O = SkipConct(FFN(LayerNorm(M)))
    """

    def __init__(self, device, dim_model, heads_en, num_q, num_c, seq_len, dropout, emb_type, emb_path="", pretrain_dim=768):
        super().__init__()
        self.seq_len = seq_len
        self.emb_path = emb_path
        self.num_c = num_c
        self.num_q = num_q
        self.device = device
        self.emb_type = emb_type
        self.emb_size = dim_model

        if num_q > 0:
            if emb_type == "qid":
                self.interaction_emb = Embedding(self.num_q, self.emb_size)
            else: 
                if emb_type == "Q_pretrain":
                    net = torch.load(emb_path)
                    self.interaction_emb = Embedding.from_pretrained(net["input_emb.weight"]) #
                else: 
                    self.interaction_emb = Embedding(self.num_q, pretrain_dim)
                    self.emb_predict = Linear(self.emb_size, 1)
                    self.drop = Dropout(0.2)
                self.emb_layer = Linear(pretrain_dim, self.emb_size) #
            
        if num_c > 0:
            self.emb_cat = nn.Embedding(num_c, embedding_dim = dim_model)
        # self.embd_pos   = nn.Embedding(seq_len, embedding_dim = dim_model)                  #positional embedding

        self.multi_en = nn.MultiheadAttention(embed_dim = dim_model, num_heads = heads_en, dropout = dropout)
        self.layer_norm1 = nn.LayerNorm(dim_model)
        self.dropout1 = Dropout(dropout)

        self.ffn_en = transformer_FFN(dim_model, dropout)
        self.layer_norm2 = nn.LayerNorm(dim_model)
        self.dropout2 = Dropout(dropout)

    def forward(self, in_ex, in_cat, in_pos, first_block=True):
        emb_type = self.emb_type
        
        ## todo create a positional encoding (two options numeric, sine)
        if first_block:
            embs = []
            if self.num_q > 0:
                if emb_type == "qid":
                    in_ex = self.interaction_emb(in_ex)
                elif emb_type == "qid_emb":
                    return self.emb_predict(self.drop(in_ex))
                else: 
                    in_ex = self.emb_layer(self.interaction_emb(in_ex))
                embs.append(in_ex)
            if self.num_c > 0:
                in_cat = self.emb_cat(in_cat)
                embs.append(in_cat)
            out = embs[0]
            for i in range(1, len(embs)):
                out += embs[i]
            out = out + in_pos
            # in_pos = self.embd_pos(in_pos)
        else:
            out = in_ex
        
        # in_pos = get_pos(self.seq_len)
        # in_pos = self.embd_pos(in_pos)

        out = out.permute(1,0,2)                                # (n,b,d)  # print('pre multi', out.shape)
        
        # norm -> attn -> drop -> skip corresponging to transformers' norm_first
        #Multihead attention                            
        n,_,_ = out.shape
        out = self.layer_norm1(out)                           # Layer norm
        skip_out = out 
        out, attn_wt = self.multi_en(out, out, out,
                                attn_mask=ut_mask(self.device, seq_len=n))  # attention mask upper triangular
        out = self.dropout1(out)
        out = out + skip_out                                    # skip connection

        #feed forward
        out = out.permute(1,0,2)                                # (b,n,d)
        out = self.layer_norm2(out)                           # Layer norm 
        skip_out = out
        out = self.ffn_en(out)
        out = self.dropout2(out)
        out = out + skip_out                                    # skip connection

        return out


class Decoder_block(nn.Module):
    """
    M1 = SkipConct(Multihead(LayerNorm(Qin;Kin;Vin)))
    M2 = SkipConct(Multihead(LayerNorm(M1;O;O)))
    L = SkipConct(FFN(LayerNorm(M2)))
    """

    def __init__(self, device, dim_model, total_res, heads_de, seq_len, dropout, emb_type):
        super().__init__()
        self.seq_len    = seq_len
        # self.embd_pos   = nn.Embedding(seq_len, embedding_dim = dim_model)                  #positional embedding
        self.multi_de1  = nn.MultiheadAttention(embed_dim= dim_model, num_heads= heads_de, dropout=dropout)  # M1 multihead for interaction embedding as q k v
        self.multi_de2  = nn.MultiheadAttention(embed_dim= dim_model, num_heads= heads_de, dropout=dropout)  # M2 multihead for M1 out, encoder out, encoder out as q k v
        self.ffn_en     = transformer_FFN(dim_model, dropout)                                         # feed forward layer

        self.layer_norm1 = nn.LayerNorm(dim_model)
        self.layer_norm2 = nn.LayerNorm(dim_model)
        self.layer_norm3 = nn.LayerNorm(dim_model)

        self.dropout1 = Dropout(dropout)
        self.dropout2 = Dropout(dropout)
        self.dropout3 = Dropout(dropout)

        self.device = device 
        self.emb_type = emb_type
        self.emb_size = dim_model

        if emb_type.startswith("R_"):
            self.token_num = int(emb_type.split("_")[-1])
            self.embd_res = Embedding(self.token_num*2+1, self.emb_size)
        else:
            self.embd_res = nn.Embedding(total_res+1, self.emb_size) #response embedding, include a start token
        
        if emb_type.startswith("R_"):
            self.token_num = int(emb_type.split("_")[-1])
            if emb_type.startswith("R_dadd"):
                self.diff_emb = Embedding(self.token_num+1, self.emb_size)
                self.embd_res = Embedding(2+1, self.emb_size) #
            elif emb_type.startswith("R_sinu"): 
                diff_vec = torch.from_numpy(self.get_sinusoid_encoding_table(self.token_num*2+1, self.emb_size)).to(device)
                self.embd_res = Embedding.from_pretrained(diff_vec, freeze=False)
            else:
                self.embd_res = Embedding(self.token_num*2+1, self.emb_size)

    def get_sinusoid_encoding_table(self, n_seq, d_hidn):
        def cal_angle(position, i_hidn):
            return position / np.power(10000, 2 * (i_hidn // 2) / d_hidn)
        def get_posi_angle_vec(position):
            return [cal_angle(position, i_hidn) for i_hidn in range(d_hidn)]

        sinusoid_table = np.array([get_posi_angle_vec(i_seq) for i_seq in range(n_seq)])
        sinusoid_table[:, 0::2] = np.sin(sinusoid_table[:, 0::2])  # even index sin 
        sinusoid_table[:, 1::2] = np.cos(sinusoid_table[:, 1::2])  # odd index cos

        return sinusoid_table


    def forward(self, diff, in_res, in_pos, en_out, first_block=True):
        emb_type = self.emb_type
         ## todo create a positional encoding (two options numeric, sine)
        if first_block:
            if emb_type.startswith("R_"):
                if emb_type.startswith("R_dadd"):
                    start_token = torch.tensor([[self.token_num]]).repeat(in_res.shape[0], 1).to(self.device)
                    diff = torch.cat((start_token, diff), dim=-1)
                    demb = self.diff_emb(diff)
                    in_in = self.embd_res(in_res)
                    in_in = in_in + demb
                else: 
                    start_token = torch.tensor([[self.token_num*2]]).repeat(in_res.shape[0], 1).to(self.device)
                    diff = torch.cat((start_token, diff), dim=-1)
                    diff_x = diff + self.token_num
                    diff_ox = torch.where(in_res == 1 , diff.long(), diff_x.long()) # [batch, length]
                    diff_ox = torch.where(in_res == 2 , diff.long(), diff_ox.long()) # [batch, length]
                    in_in = self.embd_res(diff_ox).float()
            else: 
                in_in = self.embd_res(in_res)
            #combining the embedings
            out = in_in + in_pos                         # (b,n,d)
        else:
            out = in_res

        # in_pos = get_pos(self.seq_len)
        # in_pos = self.embd_pos(in_pos)

        out = out.permute(1,0,2)                                    # (n,b,d)# print('pre multi', out.shape)
        n,_,_ = out.shape

        #Multihead attention M1                                     ## todo verify if E to passed as q,k,v
        out = self.layer_norm1(out)
        skip_out = out
        out, attn_wt = self.multi_de1(out, out, out, 
                                     attn_mask=ut_mask(self.device, seq_len=n)) # attention mask upper triangular
        out = self.dropout1(out)
        out = skip_out + out                                        # skip connection

        #Multihead attention M2                                     ## todo verify if E to passed as q,k,v
        en_out = en_out.permute(1,0,2)                              # (b,n,d)-->(n,b,d)
        en_out = self.layer_norm2(en_out)
        skip_out = out
        out, attn_wt = self.multi_de2(out, en_out, en_out,
                                    attn_mask=ut_mask(self.device, seq_len=n))  # attention mask upper triangular
        out = self.dropout2(out)
        out = out + skip_out

        #feed forward
        out = out.permute(1,0,2)                                    # (b,n,d)
        out = self.layer_norm3(out)                               # Layer norm 
        skip_out = out
        out = self.ffn_en(out)                                    
        out = self.dropout3(out)
        out = out + skip_out                                        # skip connection

        return out