# -*- coding: utf-8 -*-
# Author: ???
import math
import warnings

import torch
import torch.nn as nn
import torch.nn.functional as F

warnings.filterwarnings('ignore')
import copy


class PositionalEmbedding(nn.Module):
    def __init__(self, device=None):
        super().__init__()
        self.device=device

    def forward(self, x):
        d_model, max_len = x.size()[-1], x.size()[-2]
        # Compute the positional encodings once in log space.
        pe = torch.zeros(max_len, d_model).float().to(self.device)
        pe.require_grad = False

        position = torch.arange(0, max_len).float().unsqueeze(1).to(self.device)
        div_term = (torch.arange(0, d_model, 2).float() * -(math.log(10000.0) / d_model)).exp().to(self.device)

        pe[:, 0::2] = torch.sin(position * div_term)
        pe[:, 1::2] = torch.cos(position * div_term)

        pe = pe.unsqueeze(0)
        return x+pe


class Attention(nn.Module):
    def forward(self, query, key, value, mask=None, dropout=None):
        scores = torch.matmul(query, key.transpose(-2, -1)) \
                 / math.sqrt(query.size(-1))
        if mask is not None:
            scores = scores.masked_fill(mask == 0, -1e9)

        p_attn = F.softmax(scores, dim=-1)

        if dropout is not None:
            p_attn = dropout(p_attn)

        return torch.matmul(p_attn, value), p_attn


class MultiHeadedAttention(nn.Module):
    def __init__(self, h, d_model, dropout=0.1):
        super().__init__()
        assert d_model % h == 0
        # We assume d_v always equals d_k
        self.d_k = d_model // h
        self.h = h
        self.output_linear = nn.Linear(d_model, d_model, bias=False)
        self.attention = Attention()
        self.dropout = nn.Dropout(p=dropout)

    def forward(self, query, key, value, mask=None):
        batch_size = query.size(0)
        query = query.view(batch_size, -1, self.h, self.d_k).transpose(1, 2)
        key = key.view(batch_size, -1, self.h, self.d_k).transpose(1, 2)
        value = value.view(batch_size, -1, self.h, self.d_k).transpose(1, 2)
        x, attn = self.attention(query, key, value, mask=mask, dropout=self.dropout)
        x = x.transpose(1, 2).contiguous().view(batch_size, -1, self.h * self.d_k)
        return self.output_linear(x)


class FindKVblock(nn.Module):
    def __init__(self,
                 hidden_dim=128,
                 output_dim=128,
                 n_stages=0,
                 n_steps=0,
                 device=None
                 ):
        super(FindKVblock, self).__init__()
        self.n_stages = n_stages
        self.n_steps = n_steps
        self.hidden_dim = hidden_dim
        self.output_dim = output_dim
        self.device = device
        self.block_level_att = Attention()
        self.step_level_att = Attention()
        self.fusion_lin = torch.nn.Linear(in_features=2*self.hidden_dim, out_features=self.hidden_dim)

    def forward(self, x):
        x_size = x.size()
        n_batchsize, n_timestep, n_orifeatdim = x_size[0], x_size[1], x_size[2]
        if n_timestep % self.n_stages != 0:
            post_pad_len = int(n_timestep / self.n_stages) - (self.n_timestep % self.n_stages)
            self.n_stages = self.n_stages + 1
            x = torch.nn.functional.pad(input=x, pad=(0, 0, 0, post_pad_len), mode='constant', value=0.0)
        chunked_x = torch.reshape(x, [n_batchsize*self.n_stages, int(n_timestep/self.n_stages), n_orifeatdim])
        chunked_x_q = torch.mean(chunked_x, dim=1, keepdim=True)
        chunked_x_q = torch.reshape(chunked_x_q, [n_batchsize ,self.n_stages, n_orifeatdim])
        chunked_x_q = self.block_level_att(chunked_x_q, chunked_x_q, chunked_x_q)[0]
        chunked_x_q_block = chunked_x_q
        chunked_x_q = torch.reshape(chunked_x_q, [n_batchsize*self.n_stages, 1, n_orifeatdim])
        chunked_x_q_step = self.step_level_att(chunked_x_q, chunked_x, chunked_x)[0]
        chunked_x_q_step = torch.reshape(chunked_x_q_step, [n_batchsize, self.n_stages, n_orifeatdim])
        chunked_x_q = self.fusion_lin(torch.cat([chunked_x_q_block,chunked_x_q_step],dim=-1))
        return chunked_x_q


class StageEncoderLayer(nn.Module):
    def __init__(self,
                 d_model=128,
                 nhead=8,
                 device=None
                 ):
        super(StageEncoderLayer, self).__init__()
        # assert input_size != None and isinstance(input_size, int), 'fill in correct input_size'
        self.hidden_dim = d_model
        self.nhead = nhead
        self.device = device
        self.qkvatt = MultiHeadedAttention(d_model=self.hidden_dim, h=self.nhead, dropout=0.5)
        self.mlp1 = nn.Linear(self.hidden_dim, self.hidden_dim)
        self.norm1 = torch.nn.LayerNorm(normalized_shape=self.hidden_dim)
        self.norm2 = torch.nn.LayerNorm(normalized_shape=self.hidden_dim)
        self.dropout1 = torch.nn.Dropout(0.5)
        self.dropout2 = torch.nn.Dropout(0.5)
        self.dropout3 = torch.nn.Dropout(0.5)
        self.relu = torch.nn.ReLU()

    def forward(self, query, key, value):
        out = self.qkvatt(query=query, key=key, value=value)
        src = query + self.dropout1(out)
        src = self.norm1(src)
        src2 = self.dropout2(self.relu(self.mlp1(src)))
        src = src + self.dropout3(src2)
        src = self.norm2(src)
        return src


class StageEncoder(nn.Module):
    def __init__(self,
                 input_dim=None,
                 hidden_dim=128,
                 output_dim=128,
                 n_steps=0,
                 n_layers=1,
                 device=None
                 ):
        super(StageEncoder, self).__init__()
        self.n_stages = int(math.log(n_steps))+1
        self.n_steps = n_steps
        self.n_layers = n_layers
        self.input_size = input_dim
        self.hidden_dim = hidden_dim
        self.output_dim = output_dim
        self.device = device
        self.embed_func = torch.nn.Linear(in_features=self.input_size, out_features=self.hidden_dim)
        self.pe = PositionalEmbedding(device=self.device)
        self.find_kv_block_ = FindKVblock(hidden_dim=self.hidden_dim,
                                 output_dim=self.output_dim,
                                 n_steps=self.n_steps,
                                 n_stages=self.n_stages)
        self.encoderlayer = StageEncoderLayer(d_model=self.hidden_dim, nhead=8)
        self.core = torch.nn.ModuleList([copy.deepcopy(self.encoderlayer) for i in range(self.n_layers)])

    def forward(self, input_data):
        X = input_data[0]
        M = input_data[1]
        cur_M = input_data[2]
        n_batchsize, n_timestep, n_orifeatdim = X.shape
        assert n_timestep % self.n_stages == 0, "length of the sequence must be divisible by n_stages"
        X = self.embed_func(X)
        X = self.pe(X)
        c_temporal_k = self.find_kv_block_(X)
        c_temporal_v = c_temporal_k
        outputs = self.core[0](X, c_temporal_k, c_temporal_v)
        for i in range(1, self.n_layers, 1):
            outputs = self.core[i](outputs, c_temporal_k, c_temporal_v)
        all_output = outputs
        cur_output = (outputs * cur_M.unsqueeze(-1)).sum(dim=1)
        return all_output, cur_output
