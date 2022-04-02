#
# Created by zsd on 2022/03/10.
#
"""EET transformers clip model. """

import math
import time
import torch
import torch.nn as nn
import numpy as np
from torch import Tensor
import torch.nn.functional as F
from torch.nn.parameter import Parameter
from typing import Any, Dict, List, Optional, Tuple

from EET import MetaDesc as meta_desc
from EET import FeedForwardNetwork as eet_ffn
from EET import MultiHeadAttention as eet_attention
from EET import Embedding as eet_embedding
from EET import LayerNorm as eet_layernorm


__all__ = [
    'FCLayer', 'EETLayerNorm', 'EETFeedforward', 'EETSelfAttention', 'EETEncoderLayer', 'EETEncoder'
]


class FCLayer():
    def __init__(self, weight, bias=None, data_type=torch.float32):
        self.weight = weight.contiguous().cuda().type(data_type)
        self.bias = bias.contiguous().cuda().type(data_type) if bias else None

    def __call__(self, input):
        return F.linear(input, weight=self.weight, bias=self.bias)


class LayerNorm():
    def __init__(self, norm_shape, weight, bias, eps=1e-5, data_type=torch.float32):
        self.weight = weight.contiguous().cuda().type(data_type)
        self.bias = bias.contiguous().cuda().type(data_type)
        self.norm_shape = norm_shape
        self.eps = eps

    def __call__(self, input):
        return F.layer_norm(input, self.norm_shape, self.weight, self.bias, self.eps)


class EETLayerNorm():
    def __init__(self, config, layernorm_weight, layernorm_bias, data_type=torch.float32):
        self.layernorm_weight = layernorm_weight.cuda().type(data_type)
        self.layernorm_bias = layernorm_bias.cuda().type(data_type)
        self.layernorm = eet_layernorm(config, self.layernorm_weight, self.layernorm_bias)

    def __call__(self, input):
        return self.layernorm.layer_norm(input)
    
    @staticmethod
    def from_torch(config, layernorm_weight, layernorm_bias, data_type=torch.float32):
        layernorm = EETLayerNorm(config, layernorm_weight, layernorm_bias, data_type=data_type)
        return layernorm


class EETFeedforward():
    def __init__(self, config, model_dict, layer_id, data_type=torch.float32):
        self.intermediate_weights = torch.t(model_dict['layer.' + str(layer_id) + '.ffn.intermediate.weight']).contiguous().cuda().type(data_type)
        self.intermediate_bias = model_dict['layer.' + str(layer_id) + '.ffn.intermediate.bias'].cuda().type(data_type)
        self.output_weights = torch.t(model_dict['layer.' + str(layer_id) + '.ffn.output.weight']).contiguous().cuda().type(data_type)
        self.output_bias = model_dict['layer.' + str(layer_id) + '.ffn.output.bias'].cuda().type(data_type)
        self.layernorm_weights = model_dict['layer.' + str(layer_id) + '.ffn.layernorm.weight'].cuda().type(data_type)
        self.layernorm_bias = model_dict['layer.' + str(layer_id) + '.ffn.layernorm.bias'].cuda().type(data_type)

        self.ffn = eet_ffn(config, self.intermediate_weights, self.intermediate_bias, self.output_weights, self.output_bias, self.layernorm_weights, self.layernorm_bias)

    def __call__(
        self,
        hidden_states,
        pre_layernorm=True,
        add_redusial=True
    ):
        return self.ffn.forward(hidden_states, pre_layernorm, add_redusial)

    @staticmethod
    def from_torch(config, model_dict, layer_id, data_type=torch.float32):
        feedforward = EETFeedforward(config, model_dict, layer_id, data_type=data_type)
        return feedforward


class EETSelfAttention():
    def __init__(self, config, model_dict, layer_id, data_type=torch.float32):
        q_weights = model_dict['layer.' + str(layer_id) + '.self_attn.q_proj.weight'].contiguous().cuda().type(data_type)
        k_weights = model_dict['layer.' + str(layer_id) + '.self_attn.k_proj.weight'].contiguous().cuda().type(data_type)
        v_weights = model_dict['layer.' + str(layer_id) + '.self_attn.v_proj.weight'].contiguous().cuda().type(data_type)
        self.qkv_weight = torch.cat((q_weights, k_weights, v_weights), 0).transpose(0, 1).contiguous()
        self.q_bias = model_dict['layer.' + str(layer_id) + '.self_attn.q_proj.bias'].cuda().type(data_type)
        self.k_bias = model_dict['layer.' + str(layer_id) + '.self_attn.k_proj.bias'].cuda().type(data_type)
        self.v_bias = model_dict['layer.' + str(layer_id) + '.self_attn.v_proj.bias'].cuda().type(data_type)
        self.out_weights = torch.t(model_dict['layer.' + str(layer_id) + '.self_attn.out_proj.weight']).contiguous().cuda().type(data_type)
        self.out_bias = model_dict['layer.' + str(layer_id) + '.self_attn.out_proj.bias'].cuda().type(data_type)
        self.layernorm_weights = model_dict['layer.' + str(layer_id) + '.self_attn.layernorm.weight'].cuda().type(data_type)
        self.layernorm_bias = model_dict['layer.' + str(layer_id) + '.self_attn.layernorm.bias'].cuda().type(data_type)

        self.attention = eet_attention(config, self.qkv_weight, self.q_bias, self.k_bias, self.v_bias,
                                       self.out_weights, self.out_bias, self.layernorm_weights, self.layernorm_bias)

    def __call__(
        self,
        hidden_states,
        pre_padding_len,
        pre_layernorm=False,
        add_redusial=True,
        need_sequence_mask=False,
    ):
        return self.attention.forward(hidden_states, pre_padding_len, pre_layernorm, add_redusial, need_sequence_mask)

    @staticmethod
    def from_torch(config, model_dict, layer_id, data_type=torch.float32):
        attention = EETSelfAttention(config, model_dict, layer_id, data_type=data_type)
        return attention


class EETEncoderLayer():
    def __init__(self, config, attention, feedforward):
        self.attention = attention
        self.feedforward = feedforward

    def __call__(
        self,
        hidden_states,
        pre_padding_len=None,
        normalize_before=False,
        need_sequence_mask=False
    ):

        self_attn_out = self.attention(
            hidden_states,
            pre_padding_len=pre_padding_len,
            pre_layernorm=normalize_before,
            add_redusial=True,
            need_sequence_mask=need_sequence_mask
        )
        out = self.feedforward(
            self_attn_out,
            pre_layernorm=normalize_before,
            add_redusial=True
        )
        return out

    @staticmethod
    def from_torch(config, model_dict, layer_id, data_type=torch.float32):
        attention = EETSelfAttention.from_torch(config=config, model_dict=model_dict, layer_id=layer_id, data_type=data_type)
        feedforward = EETFeedforward.from_torch(config=config, model_dict=model_dict, layer_id=layer_id, data_type=data_type)
        layer = EETEncoderLayer(config, attention, feedforward)
        return layer


class EETEncoder():
    def __init__(self, EncoderLayers):
        self.layers = EncoderLayers

    def __call__(
        self,
        hidden_states,
        pre_padding_len=None,
        normalize_before=False,
        need_sequence_mask=False,
    ):
        for layer in self.layers:
            hidden_states = layer(
                hidden_states,
                pre_padding_len=pre_padding_len,
                normalize_before=normalize_before,
                need_sequence_mask=need_sequence_mask
            )
        return hidden_states

    @staticmethod
    def from_torch(config, layer_model_dict, layer_num, data_type=torch.float32):
        """from torch."""
        EncoderLayers = []
        for i in range(layer_num):
            EncoderLayers.extend(
                [
                    EETEncoderLayer.from_torch(config, layer_model_dict['layer.' + str(i)], i, data_type=data_type)
                ]
            )
        eet_encoder = EETEncoder(EncoderLayers)
        return eet_encoder




