#
# Created by zsd on 2022/03/10.
#
"""EET transformers encoder model. """

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
from EET import CrossMultiHeadAttention as eet_cross_attention
from EET import MaskedMultiHeadAttention as eet_masked_attention
from EET import Embedding as eet_embedding
from EET import LayerNorm as eet_layernorm


__all__ = [
    'FCLayer', 'EETLayerNorm', 'EETFeedforward', 'EETSelfAttention', 'EETEncoderLayer', 'EETEncoder', 
    'EETSelfMaskedAttention', 'EETCrossAttention', 'EETDecoderLayer', 'EETDecoder',
]


class FCLayer():
    def __init__(self, weight, bias=None, data_type=torch.float32):
        self.weight = weight.contiguous().cuda().type(data_type)
        self.bias = bias.contiguous().cuda().type(data_type) if bias is not None else None

    def __call__(self, input):
        return F.linear(input, weight=self.weight, bias=self.bias)


class EETLayerNorm():
    r"""
    EET implementation of layernorm
    Applies Layer Normalization over a mini-batch of inputs, same as torch.nn.LayerNorm
    Args:


    Returns:

    Example:
    ```python
    >>> from eet.transformers.encoder import EETLayerNorm
    >>> 
    """
    def __init__(self, config, layernorm_weight, layernorm_bias, data_type=torch.float32):
        self.layernorm_weight = layernorm_weight.cuda().type(data_type)
        self.layernorm_bias = layernorm_bias.cuda().type(data_type) if layernorm_bias is not None else torch.empty(0)
        self.layernorm = eet_layernorm(config, self.layernorm_weight, self.layernorm_bias)

    def __call__(self, input):
        return self.layernorm.layer_norm(input)
    
    @staticmethod
    def from_torch(config, layernorm_weight, layernorm_bias, data_type=torch.float32):
        layernorm = EETLayerNorm(config, layernorm_weight, layernorm_bias, data_type=data_type)
        return layernorm


class EETFeedforward():
    r"""

    
    """
    def __init__(self, config, model_dict, layer_id, data_type=torch.float32, bias=True, name="out_cache"):
        self.intermediate_weights = torch.t(model_dict['layer.' + str(layer_id) + '.ffn.intermediate.weight']).contiguous().cuda().type(data_type)
        self.intermediate_bias = model_dict['layer.' + str(layer_id) + '.ffn.intermediate.bias'].cuda().type(data_type) if bias else torch.empty(0)
        self.output_weights = torch.t(model_dict['layer.' + str(layer_id) + '.ffn.output.weight']).contiguous().cuda().type(data_type)
        self.output_bias = model_dict['layer.' + str(layer_id) + '.ffn.output.bias'].cuda().type(data_type) if bias else torch.empty(0)
        self.layernorm_weights = model_dict['layer.' + str(layer_id) + '.ffn.layernorm.weight'].cuda().type(data_type)
        self.layernorm_bias = model_dict['layer.' + str(layer_id) + '.ffn.layernorm.bias'].cuda().type(data_type) if bias else torch.empty(0)

        self.ffn = eet_ffn(config, self.intermediate_weights, self.intermediate_bias, self.output_weights, self.output_bias, self.layernorm_weights, self.layernorm_bias, name)

    def __call__(
        self,
        hidden_states,
        pre_layernorm=True,
        add_redusial=True
    ):
        return self.ffn.forward(hidden_states, pre_layernorm, add_redusial)

    @staticmethod
    def from_torch(config, model_dict, layer_id, data_type=torch.float32, bias=True, name="out_cache"):
        feedforward = EETFeedforward(config, model_dict, layer_id, data_type=data_type, bias=bias, name=name)
        return feedforward


class EETSelfAttention():
    r"""
        This is an EET implementation of 
        is_standard: q_proj, k_proj, v_proj = Linear; qkv_proj = Conv1D
    """
    def __init__(self, config, model_dict, layer_id, data_type=torch.float32, bias=True, is_standard=True):
        self.is_standard = is_standard
        self.layernorm_weights = model_dict['layer.' + str(layer_id) + '.self_attn.layernorm.weight'].cuda().type(data_type)
        self.layernorm_bias = model_dict['layer.' + str(layer_id) + '.self_attn.layernorm.bias'].cuda().type(data_type) if bias else torch.empty(0)
        emb_size = self.layernorm_weights.size()[-1]

        if self.is_standard:
            q_weights = model_dict['layer.' + str(layer_id) + '.self_attn.q_proj.weight'].contiguous().cuda().type(data_type)
            k_weights = model_dict['layer.' + str(layer_id) + '.self_attn.k_proj.weight'].contiguous().cuda().type(data_type)
            v_weights = model_dict['layer.' + str(layer_id) + '.self_attn.v_proj.weight'].contiguous().cuda().type(data_type)
            self.qkv_weight = torch.cat((q_weights, k_weights, v_weights), 0).transpose(0, 1).contiguous()
            self.q_bias = model_dict['layer.' + str(layer_id) + '.self_attn.q_proj.bias'].cuda().type(data_type) if bias else torch.empty(0)
            self.k_bias = model_dict['layer.' + str(layer_id) + '.self_attn.k_proj.bias'].cuda().type(data_type) if bias else torch.empty(0)
            self.v_bias = model_dict['layer.' + str(layer_id) + '.self_attn.v_proj.bias'].cuda().type(data_type) if bias else torch.empty(0)
            self.out_weights = torch.t(model_dict['layer.' + str(layer_id) + '.self_attn.out_proj.weight']).contiguous().cuda().type(data_type)
            self.out_bias = model_dict['layer.' + str(layer_id) + '.self_attn.out_proj.bias'].cuda().type(data_type) if bias else torch.empty(0)
        else:
            self.qkv_weight = model_dict['layer.' + str(layer_id) + '.self_attn.qkv_proj.weight'].contiguous().cuda().type(data_type)
            self.q_bias = model_dict['layer.' + str(layer_id) + '.self_attn.qkv_proj.bias'][:emb_size].cuda().type(data_type) if bias else torch.empty(0)
            self.k_bias = model_dict['layer.' + str(layer_id) + '.self_attn.qkv_proj.bias'][emb_size:emb_size*2].cuda().type(data_type) if bias else torch.empty(0)
            self.v_bias = model_dict['layer.' + str(layer_id) + '.self_attn.qkv_proj.bias'][emb_size*2:].cuda().type(data_type) if bias else torch.empty(0)
            self.out_weights = model_dict['layer.' + str(layer_id) + '.self_attn.out_proj.weight'].contiguous().cuda().type(data_type)
            self.out_bias = model_dict['layer.' + str(layer_id) + '.self_attn.out_proj.bias'].cuda().type(data_type) if bias else torch.empty(0)

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
    def from_torch(config, model_dict, layer_id, data_type=torch.float32, bias=True, is_standard=True):
        attention = EETSelfAttention(config, model_dict, layer_id, data_type=data_type, bias=bias, is_standard=is_standard)
        return attention


class EETCrossAttention():
    def __init__(self, config, model_dict, layer_id, data_type=torch.float32, bias=True, is_standard=True):
        self.is_standard = is_standard
        self.layernorm_weights = model_dict['layer.' + str(layer_id) + '.encoder_attn.layernorm.weight'].cuda().type(data_type)
        self.layernorm_bias = model_dict['layer.' + str(layer_id) + '.encoder_attn.layernorm.bias'].cuda().type(data_type) if bias else torch.empty(0)
        emb_size = self.layernorm_weights.size()[-1]

        if self.is_standard:
            self.q_weights = torch.t(model_dict['layer.' + str(layer_id) + '.encoder_attn.q_proj.weight']).contiguous().cuda().type(data_type)
            self.k_weights = torch.t(model_dict['layer.' + str(layer_id) + '.encoder_attn.k_proj.weight']).contiguous().cuda().type(data_type)
            self.v_weights = torch.t(model_dict['layer.' + str(layer_id) + '.encoder_attn.v_proj.weight']).contiguous().cuda().type(data_type)
            self.q_bias = model_dict['layer.' + str(layer_id) + '.encoder_attn.q_proj.bias'].cuda().type(data_type) if bias else torch.empty(0)
            self.k_bias = model_dict['layer.' + str(layer_id) + '.encoder_attn.k_proj.bias'].cuda().type(data_type) if bias else torch.empty(0)
            self.v_bias = model_dict['layer.' + str(layer_id) + '.encoder_attn.v_proj.bias'].cuda().type(data_type) if bias else torch.empty(0)
            self.out_weights = torch.t(model_dict['layer.' + str(layer_id) + '.encoder_attn.out_proj.weight']).contiguous().cuda().type(data_type)
            self.out_bias = model_dict['layer.' + str(layer_id) + '.encoder_attn.out_proj.bias'].cuda().type(data_type) if bias else torch.empty(0)
        else:
            self.q_weights = model_dict['layer.' + str(layer_id) + '.encoder_attn.q_proj.weight'].contiguous().cuda().type(data_type)
            self.k_weights = model_dict['layer.' + str(layer_id) + '.encoder_attn.kv_proj.weight'][:, :emb_size].contiguous().cuda().type(data_type)
            self.v_weights = model_dict['layer.' + str(layer_id) + '.encoder_attn.kv_proj.weight'][:, emb_size:].contiguous().cuda().type(data_type)
            self.q_bias = model_dict['layer.' + str(layer_id) + '.encoder_attn.q_proj.bias'].cuda().type(data_type) if bias else torch.empty(0)
            self.k_bias = model_dict['layer.' + str(layer_id) + '.encoder_attn.kv_proj.bias'][:emb_size].cuda().type(data_type) if bias else torch.empty(0)
            self.v_bias = model_dict['layer.' + str(layer_id) + '.encoder_attn.kv_proj.bias'][emb_size:].cuda().type(data_type) if bias else torch.empty(0)
            self.out_weights = model_dict['layer.' + str(layer_id) + '.encoder_attn.out_proj.weight'].contiguous().cuda().type(data_type)
            self.out_bias = model_dict['layer.' + str(layer_id) + '.encoder_attn.out_proj.bias'].cuda().type(data_type) if bias else torch.empty(0)

        self.attetion = eet_cross_attention(config, self.q_weights, self.k_weights, self.v_weights, self.q_bias,
                                            self.k_bias, self.v_bias, self.out_weights, self.out_bias, self.layernorm_weights, self.layernorm_bias)

    def __call__(
        self,
        input_id,
        pre_padding_len,
        encoder_out=None,
        encoder_padding_mask=None,
        pre_layernorm=False,
        add_redusial=True,
        first_pass=False
    ):
        # print('eet encoder out shape: ', encoder_out.shape)
        return self.attetion.forward(input_id, encoder_out, pre_padding_len, pre_layernorm, add_redusial, encoder_padding_mask, first_pass)

    @staticmethod
    def from_torch(config, model_dict, layer_id, data_type=torch.float32, bias=True, is_standard=True):
        attention = EETCrossAttention(config, model_dict, layer_id, data_type=data_type, bias=bias, is_standard=is_standard)
        return attention


class EETSelfMaskedAttention():
    def __init__(self, config, model_dict, layer_id, data_type=torch.float32, bias=True, is_standard=True):
        self.is_standard = is_standard
        self.layernorm_weights = model_dict['layer.' + str(layer_id) + '.self_attn.layernorm.weight'].cuda().type(data_type)
        self.layernorm_bias = model_dict['layer.' + str(layer_id) + '.self_attn.layernorm.bias'].cuda().type(data_type) if bias else torch.empty(0)
        emb_size = self.layernorm_weights.size()[-1]

        if self.is_standard:
            q_weights = model_dict['layer.' + str(layer_id) + '.self_attn.q_proj.weight'].contiguous().cuda().type(data_type)
            k_weights = model_dict['layer.' + str(layer_id) + '.self_attn.k_proj.weight'].contiguous().cuda().type(data_type)
            v_weights = model_dict['layer.' + str(layer_id) + '.self_attn.v_proj.weight'].contiguous().cuda().type(data_type)
            self.qkv_weight = torch.cat((q_weights, k_weights, v_weights), 0).transpose(0, 1).contiguous()
            self.q_bias = model_dict['layer.' + str(layer_id) + '.self_attn.q_proj.bias'].cuda().type(data_type) if bias else torch.empty(0)
            self.k_bias = model_dict['layer.' + str(layer_id) + '.self_attn.k_proj.bias'].cuda().type(data_type) if bias else torch.empty(0)
            self.v_bias = model_dict['layer.' + str(layer_id) + '.self_attn.v_proj.bias'].cuda().type(data_type) if bias else torch.empty(0)
            self.out_weights = torch.t(model_dict['layer.' + str(layer_id) + '.self_attn.out_proj.weight']).contiguous().cuda().type(data_type)
            self.out_bias = model_dict['layer.' + str(layer_id) + '.self_attn.out_proj.bias'].cuda().type(data_type) if bias else torch.empty(0)
        else:
            self.qkv_weight = model_dict['layer.' + str(layer_id) + '.self_attn.qkv_proj.weight'].contiguous().cuda().type(data_type)
            self.q_bias = model_dict['layer.' + str(layer_id) + '.self_attn.qkv_proj.bias'][:emb_size].cuda().type(data_type) if bias else torch.empty(0)
            self.k_bias = model_dict['layer.' + str(layer_id) + '.self_attn.qkv_proj.bias'][emb_size:emb_size*2].cuda().type(data_type) if bias else torch.empty(0)
            self.v_bias = model_dict['layer.' + str(layer_id) + '.self_attn.qkv_proj.bias'][emb_size*2:].cuda().type(data_type) if bias else torch.empty(0)
            self.out_weights = model_dict['layer.' + str(layer_id) + '.self_attn.out_proj.weight'].contiguous().cuda().type(data_type)
            self.out_bias = model_dict['layer.' + str(layer_id) + '.self_attn.out_proj.bias'].cuda().type(data_type) if bias else torch.empty(0)

        self.attetion = eet_masked_attention(config, self.qkv_weight, self.q_bias, self.k_bias,
                                             self.v_bias, self.out_weights, self.out_bias, self.layernorm_weights, self.layernorm_bias)

    def __call__(
        self,
        input_id,
        pre_padding_len,
        reorder_state=None,
        pre_layernorm=False,
        add_redusial=True,
        first_pass=False
    ):
        return self.attetion.forward(input_id, pre_padding_len, reorder_state, pre_layernorm, add_redusial, first_pass)

    @staticmethod
    def from_torch(config, model_dict, layer_id, data_type=torch.float32, bias=True, is_standard=True):
        attention = EETSelfMaskedAttention(config, model_dict, layer_id, data_type=data_type, bias=bias, is_standard=is_standard)
        return attention


class EETEncoderLayer():
    r"""
        EET Encoder Layer. 
        Args:

        Inputs:

        Ouptuts:


        Examples:

    """
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
    def from_torch(config, model_dict, layer_id, data_type=torch.float32, bias=True, name='encoder_out_cache'):
        attention = EETSelfAttention.from_torch(config=config, model_dict=model_dict, layer_id=layer_id, data_type=data_type, bias=bias)
        feedforward = EETFeedforward.from_torch(config=config, model_dict=model_dict, layer_id=layer_id, data_type=data_type, bias=bias, name=name)
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
    def from_torch(config, layer_model_dict, layer_num, data_type=torch.float32, bias=True):
        """from torch."""
        EncoderLayers = []
        for i in range(layer_num):
            EncoderLayers.extend(
                [
                    EETEncoderLayer.from_torch(config, layer_model_dict['layer.' + str(i)], i, data_type=data_type, bias=bias)
                ]
            )
        eet_encoder = EETEncoder(EncoderLayers)
        return eet_encoder


class EETDecoderLayer():
    def __init__(self, config, attention, feedforward, cross_attention=None):
        self.attetion = attention
        self.cross_attention = cross_attention
        self.feedforward = feedforward
    
    def __call__(
        self,
        x,
        encoder_out=None,
        first_pass=True,
        pre_padding_len=None,
        encoder_attention_mask=None,
        head_mask=None,
        reorder_state=None,
        normalize_before=True,
        add_redusial=True,
    ):

        if encoder_out is not None and self.cross_attention is not None:
            ''' self_attn -> cross_attn -> ffn'''
            self_attn_out = self.attetion(
                input_id=x,
                pre_padding_len=pre_padding_len,
                reorder_state=reorder_state,
                pre_layernorm=normalize_before,
                add_redusial=add_redusial,
                first_pass=first_pass
            )
            # print('eet self attn out: ', self_attn_out)
            # print('eet encoder_out: ', encoder_out)
            cross_attn_out = self.cross_attention(
                input_id=self_attn_out,
                pre_padding_len=pre_padding_len,
                encoder_out=encoder_out,
                encoder_padding_mask=encoder_attention_mask,
                pre_layernorm=normalize_before,
                add_redusial=add_redusial,
                first_pass=first_pass
            )
            # print('eet cross attn out: ', cross_attn_out)
            out = self.feedforward(
                cross_attn_out,
                pre_layernorm=normalize_before,
                add_redusial=add_redusial
            )
        else:
            ''' self_attn -> ffn''' 
            self_attn_out = self.attetion(
                input_id=x,
                pre_padding_len=pre_padding_len,
                reorder_state=reorder_state,
                pre_layernorm=normalize_before,
                add_redusial=add_redusial,
                first_pass=first_pass
            )

            out = self.feedforward(
                self_attn_out,
                pre_layernorm=normalize_before,
                add_redusial=add_redusial
            )
        return out

    @staticmethod
    def from_torch(config, model_dict, layer_id, data_type=torch.float32, add_cross_attn=True, bias=True, is_standard=True):
        attention = EETSelfMaskedAttention.from_torch(config, model_dict, layer_id, data_type=data_type, bias=bias, is_standard=is_standard)
        feedforward = EETFeedforward.from_torch(config, model_dict, layer_id, data_type=data_type, bias=bias, name="decoder_out_cache")

        if add_cross_attn:
            cross_attention = EETCrossAttention.from_torch(config, model_dict, layer_id, data_type=data_type, bias=bias, is_standard=is_standard)
            layer = EETDecoderLayer(config, attention, feedforward, cross_attention)
        else:
            layer = EETDecoderLayer(config, attention, feedforward)
        
        return layer



class EETDecoder():
    def __init__(self, DecoderLayers):
        self.layers = DecoderLayers

    def __call__(
        self,
        hidden_states,
        encoder_out=None,
        first_pass=True,
        pre_padding_len=None,
        encoder_attention_mask=None,
        head_mask=None,
        reorder_state=None,
        normalize_before=False,
    ):
        for layer in self.layers:
            hidden_states = layer(
                hidden_states,
                encoder_out=encoder_out,
                first_pass=first_pass,
                pre_padding_len=pre_padding_len,
                encoder_attention_mask=encoder_attention_mask,
                head_mask=head_mask,
                reorder_state=reorder_state,
                normalize_before=normalize_before,
                add_redusial=True,
            )
        return hidden_states
    
    @staticmethod
    def from_torch(config, layer_model_dict, layer_num, data_type=torch.float32, bias=True):
        """from torch."""
        DecoderLayers = []
        for i in range(layer_num):
            DecoderLayers.extend(
                [
                    EETDecoderLayer.from_torch(config, layer_model_dict['layer.' + str(i)], i, data_type=data_type, bias=bias)
                ]
            )

        eet_decoder = EETDecoder(DecoderLayers)
        return eet_decoder