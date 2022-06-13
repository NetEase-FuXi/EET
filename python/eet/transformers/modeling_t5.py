#
# Created by zsd on 2022/04/20.
#
"""EET T5 model. """

import math
import time
import copy
import torch
import torch.nn as nn
import numpy as np
from torch import Tensor
import torch.nn.functional as F
from torch.nn.parameter import Parameter
from typing import Any, Dict, List, Optional, Tuple
from transformers import T5Model
from eet.transformers.encoder_decoder import *
from eet.utils.mapping import convert_name

from EET import MetaDesc as meta_desc
from EET import FeedForwardNetwork as eet_ffn
from EET import MultiHeadAttention as eet_attention
from EET import CrossMultiHeadAttention as eet_cross_attention
from EET import MaskedMultiHeadAttention as eet_masked_attention
from EET import Embedding as eet_embedding
from EET import LayerNorm as eet_layernorm

__all__ = ['EETT5Embedding', 'EETT5Block', 'EETT5Encoder', 'EETT5Decoder', 'EETT5Model']


def _relative_position_bucket(relative_position, bidirectional=True, num_buckets=32, max_distance=128):
    relative_buckets = 0
    if bidirectional:
        num_buckets //= 2
        relative_buckets += (relative_position > 0).to(torch.long) * num_buckets
        relative_position = torch.abs(relative_position)
    else:
        relative_position = -torch.min(relative_position, torch.zeros_like(relative_position))
        # now relative_position is in the range [0, inf)

        # half of the buckets are for exact increments in positions
    max_exact = num_buckets // 2
    is_small = relative_position < max_exact

    # The other half of the buckets are for logarithmically bigger bins in positions up to max_distance
    relative_postion_if_large = max_exact + (
        torch.log(relative_position.float() / max_exact)
        / math.log(max_distance / max_exact)
        * (num_buckets - max_exact)
    ).to(torch.long)
    relative_postion_if_large = torch.min(
        relative_postion_if_large, torch.full_like(relative_postion_if_large, num_buckets - 1)
    )

    relative_buckets += torch.where(is_small, relative_position, relative_postion_if_large)
    return relative_buckets


class EETT5Embedding():
    def __init__(self, config, embedding_dict, data_type=torch.float32):
        self.if_layernorm = False
        self.embedding_weights = embedding_dict['embeddings.word_embeddings.weight'].cuda().type(data_type)
        # not use position_embed
        self.position_weights = torch.empty(0)
        # not use token_type
        self.token_type_ids = torch.empty(0).long()
        self.token_type_weights = torch.empty(0)
        # not use layernorm
        self.Layernorm_weights = torch.empty(0)
        self.Layernorm_bias = torch.empty(0)

        self.embedding = eet_embedding(config, self.embedding_weights, self.position_weights,
                                       self.token_type_weights, self.Layernorm_weights, self.Layernorm_bias, 'emb_cache')

    def __call__(
        self,
        input_ids,
        position_ids,
        token_type_ids=None,
    ):
        if token_type_ids is None:
            token_type_ids = self.token_type_ids
        return self.embedding.forward_transformers(input_ids, position_ids, token_type_ids, self.if_layernorm)

    @staticmethod
    def from_torch(config, embedding_dict, data_type=torch.float32):
        embedding = EETT5Embedding(config, embedding_dict, data_type=data_type)
        return embedding


class EETT5Block():
    def __init__(self, cfg, attention, feedforward, cross_attention=None, position_embedding=None, data_type=torch.float32):
        self.attention = attention
        self.cross_attention = cross_attention
        self.feedforward = feedforward
        self.position_embedding = position_embedding.cuda() if position_embedding is not None else None
        self.is_decoder = (cross_attention is not None)
        self.data_type = data_type
        self.relative_attention_num_buckets = cfg.relative_attention_num_buckets
        self.relative_attention_max_distance = cfg.relative_attention_max_distance

    def compute_bias(self, query_length, key_length):
        context_position = torch.arange(query_length, dtype=torch.long)[:, None].cuda()
        memory_position = torch.arange(key_length, dtype=torch.long)[None, :].cuda()
        relative_position = memory_position - context_position
        relative_position_bucket = _relative_position_bucket(
            relative_position,  # shape (query_length, key_length)
            bidirectional=(not self.is_decoder),
            num_buckets=self.relative_attention_num_buckets,
            max_distance=self.relative_attention_max_distance,
        )
        values = self.position_embedding(relative_position_bucket)
        values = values.permute([2, 0, 1]).unsqueeze(0)
        return values

    def __call__(
        self,
        hidden_states,
        encoder_out=None,
        first_pass=True,
        pre_padding_len=None,
        per_sample_length=None,
        head_mask=None,
        reorder_state=None,
        normalize_before=True,
        add_residual=True,
        self_past_key_values_length=0,
        position_bias=None,
    ):
        batch_size, seq_length = hidden_states.shape[:2]
        real_seq_length = seq_length + self_past_key_values_length        
        if position_bias is None and self.position_embedding is not None:
            position_bias = self.compute_bias(real_seq_length, real_seq_length)
            # if past_key_value is not none
            if self_past_key_values_length > 0:
                position_bias = position_bias[:, :, -seq_length:, :]
        if position_bias is None:
            position_bias = torch.empty(0)
        
        position_bias = position_bias.contiguous()
        # position_bias = torch.empty(0)

        if encoder_out is not None and self.cross_attention is not None:
            ''' decoder: self_masked_attn -> cross_attn -> ffn'''
            # print(position_bias.size(), 'eet pos bias: ', position_bias)
            self_attn_out = self.attention(
                hidden_states=hidden_states,
                pre_padding_len=pre_padding_len,
                reorder_state=reorder_state,
                pre_layernorm=normalize_before,
                add_residual=add_residual,
                first_pass=first_pass,
                relative_attention_bias=position_bias,
            )
            # print('eet decoder self attn out: ', self_attn_out)
            cross_attn_out = self.cross_attention(
                hidden_states=self_attn_out,
                pre_padding_len=pre_padding_len,
                encoder_out=encoder_out,
                per_sample_length=per_sample_length,
                pre_layernorm=normalize_before,
                add_residual=add_residual,
                first_pass=first_pass
            )
            # print('eet cross attn out: ', cross_attn_out)
            out = self.feedforward(
                cross_attn_out,
                pre_layernorm=normalize_before,
                add_residual=add_residual
            )
            # print('eet decoder out: ', out)
        else:
            ''' encoder: self_attn -> ffn''' 
            self_attn_out = self.attention(
                hidden_states=hidden_states,
                pre_padding_len=pre_padding_len,
                pre_layernorm=normalize_before,
                add_residual=add_residual,
                relative_attention_bias=position_bias,
            )
            # print(self_attn_out.size(), ' eet self attn out: ', self_attn_out)

            out = self.feedforward(
                self_attn_out,
                pre_layernorm=normalize_before,
                add_residual=add_residual
            )
            # print(out.size(), ' eet encoder out: ', out)
        return (out, position_bias)

    @staticmethod
    def from_torch(config, cfg, model_dict, layer_id, data_type=torch.float32, is_decoder=True, bias=True, position_embedding=None, is_standard=True):
        if is_decoder:
            attention = EETSelfMaskedAttention.from_torch(config, model_dict, layer_id, data_type=data_type, bias=bias, is_standard=True)
            cross_attention = EETCrossAttention.from_torch(config, model_dict, layer_id, data_type=data_type, bias=bias, is_standard=is_standard)
            feedforward = EETFeedforward.from_torch(config, model_dict, layer_id, data_type=data_type, bias=bias, name="decoder_out_cache")
            layer = EETT5Block(cfg, attention, feedforward, cross_attention, position_embedding=position_embedding, data_type=data_type)
        else:
            attention = EETSelfAttention.from_torch(config, model_dict, layer_id, data_type=data_type, bias=bias, is_standard=True)
            feedforward = EETFeedforward.from_torch(config, model_dict, layer_id, data_type=data_type, bias=bias, name="encoder_out_cache")
            layer = EETT5Block(cfg, attention, feedforward, position_embedding=position_embedding, data_type=data_type)
        
        return layer

class EETT5Encoder():
    def __init__(self, EncoderLayers):
        self.layers = EncoderLayers

    def __call__(
        self,
        hidden_states,
        pre_padding_len=None,
        normalize_before=False,
        position_bias=None,
    ):
        for layer in self.layers:
            (hidden_states, position_bias) = layer(
                hidden_states,
                pre_padding_len=pre_padding_len,
                normalize_before=normalize_before,
                position_bias=position_bias,
            )
        return hidden_states

    @staticmethod
    def from_torch(config, cfg, position_embedding, layer_model_dict, layer_num, data_type=torch.float32, bias=True):
        """from torch."""
        EncoderLayers = []
        for i in range(layer_num):
            EncoderLayers.extend(
                [
                    EETT5Block.from_torch(config, cfg, layer_model_dict['layer.' + str(i)], i, data_type=data_type, is_decoder=False, bias=False, position_embedding=position_embedding if i == 0 else None, is_standard=True)
                ]
            )
        eet_encoder = EETT5Encoder(EncoderLayers)
        return eet_encoder


class EETT5Decoder():
    def __init__(self, DecoderLayers):
        self.layers = DecoderLayers

    def __call__(
        self,
        hidden_states,
        encoder_out=None,
        first_pass=True,
        pre_padding_len=None,
        per_sample_length=None,
        head_mask=None,
        reorder_state=None,
        normalize_before=False,
        position_bias=None,
        self_past_key_values_length=0,
    ):
        # for layer in self.layers:
        for idx, layer in enumerate(self.layers):
            (hidden_states, position_bias) = layer(
                hidden_states,
                encoder_out=encoder_out,
                first_pass=first_pass,
                pre_padding_len=pre_padding_len,
                per_sample_length=per_sample_length,
                head_mask=None,
                reorder_state=reorder_state,
                normalize_before=normalize_before,
                add_residual=True,
                position_bias=position_bias,
                self_past_key_values_length=self_past_key_values_length,
            )

        return hidden_states

    @staticmethod
    def from_torch(config, cfg, position_embedding, layer_model_dict, layer_num, data_type=torch.float32, bias=True):
        """from torch."""
        DecoderLayers = []
        for i in range(layer_num):
            DecoderLayers.extend(
                [
                    EETT5Block.from_torch(config, cfg, layer_model_dict['layer.' + str(i)], i, data_type=data_type, is_decoder=True, bias=False, position_embedding=position_embedding if i == 0 else None, is_standard=True)
                ]
            )

        eet_decoder = EETT5Decoder(DecoderLayers)
        return eet_decoder


class EETT5Model():
    def __init__(self, cfg, shared, encoder, encoder_final_layernorm, decoder, decoder_final_layernorm):
        self.shared = shared.cuda()
        self.encoder = encoder
        self.decoder = decoder
        self.encoder_final_layernorm = encoder_final_layernorm.cuda()
        self.decoder_final_layernorm = decoder_final_layernorm.cuda()
        self.cfg = cfg
        self.pre_padding_len = torch.empty(0).long()
        self.decoder_pre_padding_len = torch.empty(0).long()
        self.reorder_state = torch.empty(0).long()

    def __call__(
        self,
        input_ids,
        encoder_out=None,
        encoder_seq_length=torch.empty(0),
        first_pass=True,
        attention_mask=None,
        decoder_input_ids=None,
        decoder_attention_mask=None,
        reorder_state=None,
        self_past_key_values_length=0,
    ):        
        if attention_mask is None:
            pre_padding_len = self.pre_padding_len
        else:
            # transformers 0 - padding;1 - nopadding
            pre_padding_len = torch.sum(1 - attention_mask, 1).int().cuda()

        if decoder_attention_mask is None:
            decoder_pre_padding_len = self.decoder_pre_padding_len
        else:
            # transformers 0 - padding;1 - nopadding
            decoder_pre_padding_len = torch.sum(1 - decoder_attention_mask, 1).int().cuda()

        per_sample_length = encoder_seq_length # encoder_seq_length是encoder output实际的长度（去掉padding len）
        if not first_pass:
            assert encoder_seq_length is not None
            encoder_out = torch.empty(0)

        if encoder_out is None:
            input_shape = input_ids.size()
            input_ids = input_ids.view(-1, input_shape[-1])
            inputs_embeds = self.shared(input_ids)
            encoder_out = self.encoder(
                hidden_states=inputs_embeds,
                pre_padding_len=pre_padding_len,
                normalize_before=True,
                position_bias=None,
            )
            encoder_out = self.encoder_final_layernorm(encoder_out)

        # print("per_sample_length: ", per_sample_length)            
        if reorder_state is not None:
            self.reorder_state = reorder_state.long()       
        # print('eet encoder out: ', encoder_out)
        if decoder_input_ids is None:
            raise ValueError(f"You have to specify decoder input ids")

        decoder_input_shape = decoder_input_ids.size()
        decoder_inputs_embeds = self.shared(decoder_input_ids)
        
        decoder_out = self.decoder(
            hidden_states=decoder_inputs_embeds,
            encoder_out=encoder_out,
            first_pass=first_pass,
            pre_padding_len=decoder_pre_padding_len,
            per_sample_length=per_sample_length,
            head_mask=None,
            reorder_state=self.reorder_state,
            normalize_before=True,
            position_bias=None,
            self_past_key_values_length=self_past_key_values_length,
        )
        decoder_out = self.decoder_final_layernorm(decoder_out)

        return decoder_out


    @staticmethod
    def from_pretrained(model_id_or_path: str, max_batch, full_seq_len, data_type=torch.float32, device_id=0):
        """from torch."""
        torch.set_grad_enabled(False)
        encoder_model_dict = {}
        decoder_model_dict = {}

        torch_model = T5Model.from_pretrained(model_id_or_path)
        model_name = type(torch_model).__name__
        cfg = torch_model.config

        for k, v in torch_model.state_dict().items():
            k = convert_name(k, model_name, verbose=False)
            if 'encoder.layer.' in k:
                k = k[k.find('layer.'):]
                encoder_model_dict[k] = v
            if 'decoder.layer.' in k:
                k = k[k.find('layer.'):]
                decoder_model_dict[k] = v

             # Group by layer id in model_dict's keys
        from itertools import groupby
        encoder_layer_model_dict = {k: dict(v) for k, v in groupby(list(encoder_model_dict.items()),
                                                                   lambda item: item[0][:(item[0].index('.', item[0].index('.')+1))])}
        decoder_layer_model_dict = {k: dict(v) for k, v in groupby(list(decoder_model_dict.items()),
                                                                   lambda item: item[0][:(item[0].index('.', item[0].index('.')+1))])}

        device = "cpu" if device_id < 0 else f"cuda:{device_id}"
        activation_fn = cfg.feed_forward_proj
        batch_size = max_batch
        encoder_config = meta_desc(batch_size, cfg.num_heads, cfg.d_model, cfg.num_layers,
                                   cfg.n_positions, cfg.n_positions, data_type, device, False,
                                   activation_fn)
        decoder_config = meta_desc(batch_size, cfg.num_heads, cfg.d_model, cfg.num_decoder_layers,
                                   cfg.n_positions, cfg.n_positions, data_type, device, False,
                                   activation_fn)

        if data_type==torch.float16:
            torch_model = torch_model.half()
        else:
            torch_model = torch_model.float()

        shared = torch_model.shared
        encoder_final_layernorm = torch_model.encoder.final_layer_norm
        decoder_final_layernorm = torch_model.decoder.final_layer_norm
        encoder_position_embedding = torch_model.encoder.block[0].layer[0].SelfAttention.relative_attention_bias
        decoder_position_embedding = torch_model.decoder.block[0].layer[0].SelfAttention.relative_attention_bias

        encoder = EETT5Encoder.from_torch(encoder_config, cfg, encoder_position_embedding, encoder_layer_model_dict, cfg.num_layers, data_type=data_type, bias=False)
        decoder = EETT5Decoder.from_torch(decoder_config, cfg, decoder_position_embedding, decoder_layer_model_dict, cfg.num_decoder_layers, data_type=data_type, bias=False)
        eet_model = EETT5Model(cfg, shared, encoder, encoder_final_layernorm, decoder, decoder_final_layernorm)

        return eet_model