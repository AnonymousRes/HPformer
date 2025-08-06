# Copyright (c) Meta Platforms, Inc. and affiliates.
# All rights reserved.

# This source code is licensed under the license found in the
# LICENSE file in the root directory of this source tree.
from typing import Callable, Optional, Tuple, Union, List
import math
import torch
import torch.nn as nn
import torch.nn.functional as F
from fairseq import utils
from fairseq.modules.sequence_norm import SequenceNorm

from fairseq.modules import (
    FairseqDropout,
    LayerDropModuleList,
    LayerNorm,
    PositionalEmbedding,
    RealNumberEmbedding
)
from fairseq.modules.quant_noise import quant_noise as apply_quant_noise_
from fairseq.modules.quant_noise import quant_noise


def init_bert_params(module):
    """
    Initialize the weights specific to the BERT Model.
    This overrides the default initializations depending on the specified arguments.
        1. If normal_init_linear_weights is set then weights of linear
           layer will be initialized using the normal distribution and
           bais will be set to the specified value.
        2. If normal_init_embed_weights is set then weights of embedding
           layer will be initialized using the normal distribution.
        3. If normal_init_proj_weights is set then weights of
           in_project_weight for MultiHeadAttention initialized using
           the normal distribution (to be validated).
    """

    if isinstance(module, nn.Linear):
        module.weight.data.normal_(mean=0.0, std=0.02)
        if module.bias is not None:
            module.bias.data.zero_()
    if isinstance(module, nn.Embedding):
        module.weight.data.normal_(mean=0.0, std=0.02)
        if module.padding_idx is not None:
            module.weight.data[module.padding_idx].zero_()


class Attention(nn.Module):
    def forward(self, query, key, value, self_attn_padding_mask=None, dropout=None):
        scores = torch.matmul(query, key.transpose(-2, -1)) \
                 / math.sqrt(query.size(-1))
        # print('query.shape', query.shape)
        # print('key.shape', key.shape)
        # print('value.shape', value.shape)
        # print('scores.shape', scores.shape)
        # if self_attn_padding_mask is not None:
        #     scores = scores.masked_fill(self_attn_padding_mask == 0, -1e9)

        p_attn = F.softmax(scores, dim=-1)

        return torch.matmul(p_attn, value), p_attn


class MultiheadAttention(nn.Module):
    def __init__(self, embedding_dim, num_attention_heads, dropout=0.1):
        super().__init__()
        assert embedding_dim % num_attention_heads == 0
        self.embedding_dim = embedding_dim
        self.num_attention_heads = num_attention_heads
        self.dropout = dropout
        self.d_k = self.embedding_dim // self.num_attention_heads # We assume d_v always equals d_k
        self.output_linear = nn.Linear(self.embedding_dim, self.embedding_dim, bias=False)
        self.attention = Attention()
        self.att_dropout = nn.Dropout(p=dropout)

    def forward(self, query, key, value, self_attn_padding_mask=None):
        batch_size = query.size(0)
        # q_len = query.size(1)
        # k_len = key.size(1)
        query = query.view(batch_size, -1, self.num_attention_heads, self.d_k).transpose(1, 2)
        key = key.view(batch_size, -1, self.num_attention_heads, self.d_k).transpose(1, 2)
        value = value.view(batch_size, -1, self.num_attention_heads, self.d_k).transpose(1, 2)
        # query = self.rope(query, q_len)
        # key = self.rope(key, k_len)
        x, attn = self.attention(query, key, value, self_attn_padding_mask=self_attn_padding_mask, dropout=self.dropout)
        x = x.transpose(1, 2).contiguous().view(batch_size, -1, self.num_attention_heads * self.d_k)
        # print('x.shape', x.shape)
        # print('attn.shape', attn.shape)
        return self.att_dropout(self.output_linear(x)), attn


class TDHPB_Layer(nn.Module):
    def __init__(self, embedding_dim=128, k_times=1):
        super(TDHPB_Layer, self).__init__()
        self.embedding_dim = embedding_dim
        self.block_level_att = Attention()
        self.step_level_att = Attention()
        self.k_times = k_times
        self.fusion_lin = torch.nn.Linear(in_features=2*self.embedding_dim, out_features=self.embedding_dim)
    def forward(self, x):
        x_size = x.size()
        n_batchsize, n_timestep, n_orifeatdim = x_size[0], x_size[1], x_size[2]
        n_stages = (int(math.log(n_timestep))+1) * self.k_times
        block_size = int(n_timestep / n_stages)
        # print('block_size', block_size, 'n_stages', n_stages, 'n_timestep', n_timestep, 'n_orifeatdim', n_orifeatdim)
        if n_timestep % n_stages != 0:
            n_stages = int(n_timestep / block_size) + 1
            post_pad_len = n_stages * block_size - n_timestep
            # print('post_pad_len', post_pad_len)
            x = torch.nn.functional.pad(input=x, pad=(0, 0, 0, post_pad_len), mode='constant', value=0.0)
            # print('torch.nn.functional.pad(input=x, pad=(0, 0, 0, post_pad_len)',x.shape, block_size, n_stages)
            # exit(0)
        chunked_x = torch.reshape(x, [n_batchsize*n_stages, block_size, n_orifeatdim])
        # chunked_x_q = torch.mean(chunked_x, dim=1, keepdim=True) # task 1 2 3 4
        chunked_x_q, _ = torch.max(chunked_x, dim=1, keepdim=True) # task 5
        chunked_x_q = torch.reshape(chunked_x_q, [n_batchsize,n_stages, n_orifeatdim])
        chunked_x_q = self.block_level_att(chunked_x_q, chunked_x_q, chunked_x_q)[0]
        chunked_x_q_block = chunked_x_q
        chunked_x_q = torch.reshape(chunked_x_q, [n_batchsize*n_stages, 1, n_orifeatdim])
        # print('chunked_x_q.shape', chunked_x_q.shape)
        # print('chunked_x.shape', chunked_x.shape)
        chunked_x_q_step = self.step_level_att(chunked_x_q, chunked_x, chunked_x)[0]
        # print('chunked_x_q_step.shape', chunked_x_q_step.shape)
        chunked_x_q_step = torch.reshape(chunked_x_q_step, [n_batchsize, n_stages, n_orifeatdim])
        chunked_x_q = self.fusion_lin(torch.cat([chunked_x_q_block,chunked_x_q_step],dim=-1))
        return chunked_x_q


class HPformerSentenceEncoderLayer(nn.Module):
    """
    Implements a Transformer Encoder Layer used in BERT/XLM style pre-trained
    models.
    """
    def __init__(
        self,
        embedding_dim: int = 768,
        ffn_embedding_dim: int = 3072,
        num_attention_heads: int = 8,
        dropout: float = 0.1,
        attention_dropout: float = 0.1,
        activation_dropout: float = 0.1,
        activation_fn: str = 'relu',
        norm_type='layernorm',
        export: bool = False,
        q_noise: float = 0.0,
        qn_block_size: int = 8,
        init_fn: Callable = None,
    ) -> None:
        super().__init__()

        if init_fn is not None:
            init_fn()

        # Initialize parameters
        self.embedding_dim = embedding_dim
        self.dropout_module = FairseqDropout(dropout, module_name=self.__class__.__name__)
        self.activation_dropout_module = FairseqDropout(activation_dropout, module_name=self.__class__.__name__)

        # Initialize blocks
        self.activation_fn = activation_fn
        self.activation_fn = utils.get_activation_fn(self.activation_fn)
        self.self_attn = self.build_self_attention(
            self.embedding_dim,
            num_attention_heads,
            dropout=attention_dropout
        )

        self.norm_type = norm_type
        # layer norm associated with the self attention layer
        self.self_attn_layer_norm = SequenceNorm(self.norm_type, self.embedding_dim, affine=True, export=export) # task 5
        # self.self_attn_layer_norm = LayerNorm(self.embedding_dim, export=export) # task 1 2 3 4

        self.fc1 = self.build_fc1(
            self.embedding_dim,
            ffn_embedding_dim,
            q_noise=q_noise,
            qn_block_size=qn_block_size,
        )
        # self.fc2 = self.build_fc2(
        #     ffn_embedding_dim,
        #     self.embedding_dim,
        #     q_noise=q_noise,
        #     qn_block_size=qn_block_size,
        # )

        # layer norm associated with the position wise feed-forward NN
        # self.final_layer_norm = LayerNorm(self.embedding_dim, export=export) # task 1 2 3 4
        self.final_layer_norm = SequenceNorm(self.norm_type, self.embedding_dim, affine=True, export=export) # task 5
    def build_fc1(self, input_dim, output_dim, q_noise, qn_block_size):
        return quant_noise(
            nn.Linear(input_dim, output_dim), q_noise, qn_block_size
        )

    # def build_fc2(self, input_dim, output_dim, q_noise, qn_block_size):
    #     return quant_noise(
    #         nn.Linear(input_dim, output_dim), q_noise, qn_block_size
    #     )

    def build_self_attention(
        self,
        embed_dim,
        num_attention_heads,
        dropout
    ):
        return MultiheadAttention(
            embedding_dim=embed_dim,
            num_attention_heads=num_attention_heads,
            dropout=dropout
        )

    def forward(
        self,
        x: torch.Tensor,
        k: torch.Tensor,
        v: torch.Tensor,
        self_attn_mask: Optional[torch.Tensor] = None,
        self_attn_padding_mask: Optional[torch.Tensor] = None,
    ):
        """
        LayerNorm is applied either before or after the self-attention/ffn
        modules similar to the original Transformer implementation.
        """
        residual = x

        x, attn = self.self_attn(
            query=x,
            key=k,
            value=v,
            self_attn_padding_mask=self_attn_padding_mask
        )
        x = self.dropout_module(x)
        x = residual + x
        x = self.self_attn_layer_norm(x)

        residual = x
        x = self.activation_fn(self.fc1(x))
        x = self.activation_dropout_module(x)
        x = residual + x
        x = self.final_layer_norm(x)
        return x, attn


class HPformerLRAEncoder(nn.Module):
    """
    Implementation for a  HPformer based Sentence Encoder used
    in BERT/XLM style pre-trained models.

    Input:
        - tokens: B x T matrix representing sentences
        - segment_labels: B x T matrix representing segment label for tokens

    Output:
        - a tuple of the following:
            - a list of internal model states used to compute the
              predictions where each tensor has shape  B x T * C
            - sentence representation associated with first input token
              in format B x C.
    """

    def __init__(
            self,
            padding_idx: int,
            vocab_size: int,
            num_encoder_layers: int = 6,
            embedding_type: str = "sparse",
            embedding_dim: int = 768,
            ffn_embedding_dim: int = 3072,
            num_attention_heads: int = 8,
            ktimes: int = 1,
            dropout: float = 0.1,
            attention_dropout: float = 0.1,
            activation_dropout: float = 0.1,
            layerdrop: float = 0.0,
            max_seq_len: int = 256,
            use_position_embeddings: bool = True,
            offset_positions_by_padding: bool = True,
            encoder_normalize_before: bool = False,
            apply_bert_init: bool = False,
            activation_fn: str = "relu",
            learned_pos_embedding: bool = True,
            embed_scale: float = None,
            norm_type: str = 'layernorm',
            export: bool = False,
            traceable: bool = False,
            tie_layer_weights: bool = False,
            q_noise: float = 0.0,
            qn_block_size: int = 8,
            sen_rep_type: str = 'cls'
    ) -> None:

        super().__init__()
        self.sen_rep_type = sen_rep_type
        self.padding_idx = padding_idx
        self.vocab_size = vocab_size
        self.dropout_module = FairseqDropout(dropout, module_name=self.__class__.__name__)
        self.layerdrop = layerdrop
        self.max_seq_len = max_seq_len
        self.embedding_type = embedding_type
        self.embedding_dim = embedding_dim
        self.ktimes = ktimes
        self.use_position_embeddings = use_position_embeddings
        self.apply_bert_init = apply_bert_init
        self.learned_pos_embedding = learned_pos_embedding
        self.traceable = traceable
        self.tpu = False  # whether we're on TPU
        self.tie_layer_weights = tie_layer_weights
        self.norm_type = norm_type

        assert embedding_type in ['sparse', 'linear']
        self.embed_tokens = self.build_embedding(self.embedding_type, self.vocab_size, self.embedding_dim,
                                                 self.padding_idx)
        self.embed_scale = embed_scale

        if q_noise > 0:
            self.quant_noise = apply_quant_noise_(
                nn.Linear(self.embedding_dim, self.embedding_dim, bias=False),
                q_noise,
                qn_block_size,
            )
        else:
            self.quant_noise = None

        self.embed_positions = (
            PositionalEmbedding(
                self.max_seq_len,
                self.embedding_dim,
                padding_idx=(self.padding_idx if offset_positions_by_padding else None),
                learned=self.learned_pos_embedding,
            )
            if self.use_position_embeddings
            else None
        )
        if self.use_position_embeddings and not self.learned_pos_embedding:
            if self.embed_scale is None:
                self.embed_scale = math.sqrt(self.embedding_dim)

        self.tdhp = TDHPB_Layer(embedding_dim=self.embedding_dim, k_times=self.ktimes)
        # self.tdhp2 = TDHPB_Layer2(embedding_dim=self.embedding_dim, mx_seq=self.max_seq_len, th_dropout=dropout, k_times=self.ktimes)

        if self.layerdrop > 0.0:
            self.layers = LayerDropModuleList(p=self.layerdrop)
        else:
            self.layers = nn.ModuleList([])
        self.num_layers = num_encoder_layers
        if self.tie_layer_weights:
            real_num_layers = 1
        else:
            real_num_layers = num_encoder_layers
        self.layers.extend([
            self.build_hpformer_sentence_encoder_layer(
                embedding_dim=self.embedding_dim,
                ffn_embedding_dim=ffn_embedding_dim,
                num_attention_heads=num_attention_heads,
                dropout=self.dropout_module.p,
                attention_dropout=attention_dropout,
                activation_dropout=activation_dropout,
                activation_fn=activation_fn,
                norm_type=self.norm_type,
                export=export,
                q_noise=q_noise,
                qn_block_size=qn_block_size,
            )
            for _ in range(real_num_layers)
        ])

        if encoder_normalize_before:
            self.emb_layer_norm = LayerNorm(self.embedding_dim, export=export)
        else:
            self.emb_layer_norm = None

        # Apply initialization of model params after building the model
        if self.apply_bert_init:
            self.apply(init_bert_params)

    def build_embedding(self, embedding_type, vocab_size, embedding_dim, padding_idx):
        if embedding_type == 'sparse':
            embed_tokens = nn.Embedding(vocab_size, embedding_dim, padding_idx)
            nn.init.normal_(embed_tokens.weight, mean=0, std=embedding_dim ** -0.5)
            return embed_tokens
        else:
            embed_tokens = RealNumberEmbedding(embedding_dim)
            return embed_tokens

    def build_hpformer_sentence_encoder_layer(
            self,
            embedding_dim,
            ffn_embedding_dim,
            num_attention_heads,
            dropout,
            attention_dropout,
            activation_dropout,
            activation_fn,
            norm_type,
            export,
            q_noise,
            qn_block_size,
    ):
        return HPformerSentenceEncoderLayer(
            embedding_dim=embedding_dim,
            ffn_embedding_dim=ffn_embedding_dim,
            num_attention_heads=num_attention_heads,
            dropout=dropout,
            attention_dropout=attention_dropout,
            activation_dropout=activation_dropout,
            activation_fn=activation_fn,
            norm_type=norm_type,
            export=export,
            q_noise=q_noise,
            qn_block_size=qn_block_size,
        )

    def prepare_for_tpu_(self, **kwargs):
        self.tpu = True

    def forward(
            self,
            tokens: torch.Tensor,
            src_lengths: torch.Tensor,
            last_state_only: bool = False,
            positions: Optional[torch.Tensor] = None,
    ) -> Tuple[Union[torch.Tensor, List[torch.Tensor]], torch.Tensor]:
        # print('USING HPFORMER')
        # print('Before embedding:', tokens.shape)
        # compute padding mask. This is needed for multi-head attention
        # print(torch.max(tokens),torch.min(tokens))
        if self.embedding_type == 'sparse':
            padding_mask = tokens.eq(self.padding_idx)
            if not self.traceable and not self.tpu and not padding_mask.any():
                padding_mask = None
            # B x T -> B x T x D
            x = self.embed_tokens(tokens)
            # print('sparse embedding:', tokens.shape, x.shape)
            # print(torch.max(x), torch.min(x))
        else:
            padding_mask = None
            # B x T -> B x T x 1 -> B x T x D
            x = self.embed_tokens(tokens)
            # print('no sparse embedding:', tokens.shape, x.shape)
            # print(torch.max(x), torch.min(x))
        if self.embed_scale is not None:
            x *= self.embed_scale
            # print('embed_scale embedding:', tokens.shape, x.shape)
            # print(self.embed_scale, torch.max(x), torch.min(x))

        if self.embed_positions is not None:
            x += self.embed_positions(tokens, positions=positions)
            # print('self.embed_positions(tokens, positions=positions)', (self.embed_positions(tokens, positions=positions)).shape, x.shape)
            # print(torch.max(x), torch.min(x))
            # print('embed_positions:', tokens.shape, x.shape)
            # print('self.padding_idx:', self.padding_idx)
            # print('positions', positions)
            # print('src_lengths.shape',src_lengths.shape)
            # print('src_lengths', src_lengths)
            # print('self.vocab_size', self.vocab_size)
            # exit(0)

        if self.quant_noise is not None:
            x = self.quant_noise(x)
            # print('quant_noise:', tokens.shape, x.shape)
            # print(torch.max(x), torch.min(x))
        # assert self.emb_layer_norm is None
        if self.emb_layer_norm is not None:
            x = self.emb_layer_norm(x)
            # print('emb_layer_norm:', tokens.shape, x.shape)
            # print(torch.max(x), torch.min(x))
        x = self.dropout_module(x)
        # print('dropout_module:', tokens.shape, x.shape)
        # print(torch.max(x), torch.min(x))


        # account for padding while computing the representation
        if padding_mask is not None:
            x = x.masked_fill(padding_mask.unsqueeze(-1), 0.0)
            # print('x.masked_fill:', tokens.shape, x.shape)

        # B x T x C -> T x B x C
        # x = x.transpose(0, 1)
        # print('x:', tokens.shape, x.shape)
        # exit(0)
        c_temporal_k = self.tdhp(x)
        c_temporal_v = c_temporal_k
        # print('c_temporal_k,v:', c_temporal_k.shape, c_temporal_v.shape)
        # print(torch.max(x), torch.min(x), torch.max(c_temporal_k), torch.min(c_temporal_v))

        inner_states = []
        if not last_state_only:
            inner_states.append(x)
            # print('not last_state_only', tokens.shape, x.shape)

        for i in range(self.num_layers):
            if self.tie_layer_weights:
                j = 0
            else:
                j = i
            x, _ = self.layers[j](x=x, k=c_temporal_k, v=c_temporal_v)
            if not last_state_only:
                inner_states.append(x)

        # print('after stacked layers', x.shape)
        if padding_mask is not None:
            x = x.masked_fill(padding_mask.unsqueeze(-1), 0.0)
            # print('padding_mask', x.shape)

        if self.sen_rep_type == 'mp':
            sentence_rep = x.sum(dim=-2) / src_lengths.unsqueeze(1)
            # print('final', torch.max(x), torch.min(x), torch.max(c_temporal_k), torch.min(c_temporal_v))
            # exit(0)
            # print('mp x.sum(dim=-2).shape', x.sum(dim=-2).shape)
            # print('mp src_lengths', src_lengths)
            # print('mp src_lengths.unsqueeze(1)', src_lengths.unsqueeze(1))
            # print('mp src_lengths.unsqueeze', src_lengths.shape)
            # print('mp src_lengths.unsqueeze(1)', src_lengths.unsqueeze(1).shape)
        else:
            sentence_rep = x[0, :, :]
            # print('non mp sentence_rep', sentence_rep.shape)

        if last_state_only:
            inner_states = [x]

        if self.traceable:
            # print('self.traceable', sentence_rep.shape)
            return torch.stack(inner_states), sentence_rep
        else:
            # print('non self.traceable', sentence_rep.shape)
            return inner_states, sentence_rep
