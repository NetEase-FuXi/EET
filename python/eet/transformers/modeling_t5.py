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
from transformers import T5Model, T5ForConditionalGeneration
from transformers.modeling_outputs import (
    BaseModelOutput,
    BaseModelOutputWithPastAndCrossAttentions,
    Seq2SeqLMOutput,
    Seq2SeqModelOutput,    
)
from eet.transformers.encoder_decoder import *
from eet.utils.mapping import convert_name

from EET import MetaDesc as meta_desc
from EET import Embedding as eet_embedding
from EET import T5FeedForwardNetwork as eet_t5ffn
from ..pipelines.generation import GenerationMixin_EET
import logging
logger = logging.getLogger(__name__)


__all__ = ['EETT5Embedding', 'EETT5Block', 'EETT5Encoder', 'EETT5Decoder', 'EETT5Model', 'EETT5ForConditionalGeneration']


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

class EETT5Feedforward():
    def __init__(self, config, model_dict, layer_id, data_type=torch.float32, bias=False, name="out_cache"):
        self.intermediate_0_weights = torch.t(model_dict['layer.' + str(layer_id) + '.ffn.intermediate_0.weight']).contiguous().cuda().type(data_type)
        self.intermediate_0_bias = model_dict['layer.' + str(layer_id) + '.ffn.intermediate_0.bias'].cuda().type(data_type) if bias else torch.empty(0)
        self.intermediate_1_weights = torch.t(model_dict['layer.' + str(layer_id) + '.ffn.intermediate_1.weight']).contiguous().cuda().type(data_type)
        self.intermediate_1_bias = model_dict['layer.' + str(layer_id) + '.ffn.intermediate_1.bias'].cuda().type(data_type) if bias else torch.empty(0)
        self.output_weights = torch.t(model_dict['layer.' + str(layer_id) + '.ffn.output.weight']).contiguous().cuda().type(data_type)
        self.output_bias = model_dict['layer.' + str(layer_id) + '.ffn.output.bias'].cuda().type(data_type) if bias else torch.empty(0)
        self.layernorm_weights = model_dict['layer.' + str(layer_id) + '.ffn.layernorm.weight'].cuda().type(data_type)
        self.layernorm_bias = model_dict['layer.' + str(layer_id) + '.ffn.layernorm.bias'].cuda().type(data_type) if bias else torch.empty(0)

        self.ffn = eet_t5ffn(config, self.intermediate_0_weights, self.intermediate_0_bias, self.intermediate_1_weights, self.intermediate_1_bias,self.output_weights, self.output_bias, self.layernorm_weights, self.layernorm_bias, name)

    def __call__(
        self,
        hidden_states,
        pre_layernorm=True,
        add_residual=True
    ):
        return self.ffn.forward(hidden_states, pre_layernorm, add_residual)

    @staticmethod
    def from_torch(config, model_dict, layer_id, data_type=torch.float32, bias=True, name="out_cache"):
        feedforward = EETT5Feedforward(config, model_dict, layer_id, data_type=data_type, bias=bias, name=name)
        return feedforward

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
        encoder_outputs=None,
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

        if encoder_outputs is not None and self.cross_attention is not None:
            ''' decoder: self_masked_attn -> cross_attn -> ffn'''
            self_attn_out = self.attention(
                hidden_states=hidden_states,
                pre_padding_len=pre_padding_len,
                reorder_state=reorder_state,
                pre_layernorm=normalize_before,
                add_residual=add_residual,
                first_pass=first_pass,
                relative_attention_bias=position_bias,
            )
            cross_attn_out = self.cross_attention(
                hidden_states=self_attn_out,
                pre_padding_len=pre_padding_len,
                encoder_outputs=encoder_outputs,
                per_sample_length=per_sample_length,
                pre_layernorm=normalize_before,
                add_residual=add_residual,
                first_pass=first_pass
            )
            out = self.feedforward(
                cross_attn_out,
                pre_layernorm=normalize_before,
                add_residual=add_residual
            )
        else:
            ''' encoder: self_attn -> ffn''' 
            self_attn_out = self.attention(
                hidden_states=hidden_states,
                pre_padding_len=pre_padding_len,
                pre_layernorm=normalize_before,
                add_residual=add_residual,
                relative_attention_bias=position_bias,
            )

            out = self.feedforward(
                self_attn_out,
                pre_layernorm=normalize_before,
                add_residual=add_residual
            )
        return (out, position_bias)

    @staticmethod
    def from_torch(config, cfg, model_dict, layer_id, data_type=torch.float32, is_decoder=True, bias=True, position_embedding=None, is_standard=True):
        if is_decoder:
            attention = EETSelfMaskedAttention.from_torch(config, model_dict, layer_id, data_type=data_type, bias=bias, is_standard=True)
            cross_attention = EETCrossAttention.from_torch(config, model_dict, layer_id, data_type=data_type, bias=bias, is_standard=is_standard)
            if cfg.feed_forward_proj == "gated-gelu":
                feedforward = EETT5Feedforward.from_torch(config, model_dict, layer_id, data_type=data_type, bias=bias, name="decoder_out_cache")
            else:
                feedforward = EETFeedforward.from_torch(config, model_dict, layer_id, data_type=data_type, bias=bias, name="decoder_out_cache")
            layer = EETT5Block(cfg, attention, feedforward, cross_attention, position_embedding=position_embedding, data_type=data_type)
        else:
            attention = EETSelfAttention.from_torch(config, model_dict, layer_id, data_type=data_type, bias=bias, is_standard=True)
            if cfg.feed_forward_proj == "gated-gelu":
                feedforward = EETT5Feedforward.from_torch(config, model_dict, layer_id, data_type=data_type, bias=bias, name="encoder_out_cache")
            else:
                feedforward = EETFeedforward.from_torch(config, model_dict, layer_id, data_type=data_type, bias=bias, name="encoder_out_cache")
            layer = EETT5Block(cfg, attention, feedforward, position_embedding=position_embedding, data_type=data_type)
        
        return layer

class EETT5Encoder():
    def __init__(self, embedding, EncoderLayers, final_layer_norm):
        self.embed_tokens = embedding
        self.layers = EncoderLayers
        self.final_layer_norm = final_layer_norm

    def __call__(
        self,
        input_ids,
        pre_padding_len=None,
        normalize_before=True,
        position_bias=None,
    ):
        hidden_states = self.embed_tokens(input_ids)
        for layer in self.layers:
            (hidden_states, position_bias) = layer(
                hidden_states,
                pre_padding_len=pre_padding_len,
                normalize_before=normalize_before,
                position_bias=position_bias,
            )
        hidden_states = self.final_layer_norm(hidden_states)
        
        return hidden_states

    @staticmethod
    def from_torch(config, cfg, embedding, position_embedding, final_layer_norm, layer_model_dict, layer_num, data_type=torch.float32, bias=True):
        """from torch."""
        EncoderLayers = []
        for i in range(layer_num):
            EncoderLayers.extend(
                [
                    EETT5Block.from_torch(config, cfg, layer_model_dict['layer.' + str(i)], i, data_type=data_type, is_decoder=False, bias=False, position_embedding=position_embedding if i == 0 else None, is_standard=True)
                ]
            )
        eet_encoder = EETT5Encoder(embedding, EncoderLayers, final_layer_norm)
        return eet_encoder


class EETT5Decoder():
    def __init__(self, embedding, DecoderLayers, final_layer_norm):
        self.embed_tokens = embedding
        self.layers = DecoderLayers
        self.final_layer_norm = final_layer_norm

    def __call__(
        self,
        input_ids,
        encoder_outputs=None,
        first_pass=True,
        pre_padding_len=None,
        per_sample_length=None,
        head_mask=None,
        reorder_state=None,
        normalize_before=True,
        position_bias=None,
        self_past_key_values_length=0,
    ):
        hidden_states = self.embed_tokens(input_ids)
        for layer in self.layers:
            (hidden_states, position_bias) = layer(
                hidden_states,
                encoder_outputs=encoder_outputs,
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
        hidden_states = self.final_layer_norm(hidden_states)

        return hidden_states

    @staticmethod
    def from_torch(config, cfg, embedding, position_embedding, final_layer_norm, layer_model_dict, layer_num, data_type=torch.float32, bias=True):
        """from torch."""
        DecoderLayers = []
        for i in range(layer_num):
            DecoderLayers.extend(
                [
                    EETT5Block.from_torch(config, cfg, layer_model_dict['layer.' + str(i)], i, data_type=data_type, is_decoder=True, bias=False, position_embedding=position_embedding if i == 0 else None, is_standard=True)
                ]
            )
        eet_decoder = EETT5Decoder(embedding, DecoderLayers, final_layer_norm)
        return eet_decoder


class EETT5Model():
    def __init__(self, cfg, encoder, decoder):
        self.encoder = encoder
        self.decoder = decoder
        self.cfg = cfg
        self.pre_padding_len = torch.empty(0).long()
        self.decoder_pre_padding_len = torch.empty(0).long()
        self.reorder_state = torch.empty(0).long()

    def __call__(
        self,
        input_ids=None,
        encoder_outputs=None,
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
            encoder_outputs = torch.empty(0)

        if encoder_outputs is None:
            input_shape = input_ids.size()
            input_ids = input_ids.view(-1, input_shape[-1])
            encoder_outputs = self.encoder(
                input_ids=input_ids,
                pre_padding_len=pre_padding_len,
                normalize_before=True,
                position_bias=None,
            )

        if reorder_state is not None:
            self.reorder_state = reorder_state.long()       
        if decoder_input_ids is None:
            raise ValueError(f"You have to specify decoder input ids")
        
        decoder_out = self.decoder(
            input_ids=decoder_input_ids,
            encoder_outputs=encoder_outputs,
            first_pass=first_pass,
            pre_padding_len=decoder_pre_padding_len,
            per_sample_length=per_sample_length,
            head_mask=None,
            reorder_state=self.reorder_state,
            normalize_before=True,
            position_bias=None,
            self_past_key_values_length=self_past_key_values_length,
        )

        return decoder_out

    @staticmethod
    def from_pretrained(model_id_or_path: str, max_batch, data_type=torch.float32, device_id=0):
        """from pretrained."""
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
        if cfg.feed_forward_proj == "gated-gelu":
            activation_fn = "gelu_new"
        if not hasattr(cfg, "n_positions"):
            cfg.n_positions = 512
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

        shared = torch_model.shared.cuda()
        encoder_final_layernorm = torch_model.encoder.final_layer_norm.cuda()
        decoder_final_layernorm = torch_model.decoder.final_layer_norm.cuda()
        encoder_position_embedding = torch_model.encoder.block[0].layer[0].SelfAttention.relative_attention_bias
        decoder_position_embedding = torch_model.decoder.block[0].layer[0].SelfAttention.relative_attention_bias

        encoder = EETT5Encoder.from_torch(encoder_config, cfg, shared, encoder_position_embedding, encoder_final_layernorm, encoder_layer_model_dict, cfg.num_layers, data_type=data_type, bias=False)
        decoder = EETT5Decoder.from_torch(decoder_config, cfg, shared, decoder_position_embedding, decoder_final_layernorm, decoder_layer_model_dict, cfg.num_decoder_layers, data_type=data_type, bias=False)
        eet_model = EETT5Model(cfg, encoder, decoder)

        return eet_model

    @staticmethod
    def from_torch(torch_model, max_batch, data_type=torch.float32):
        """from torch"""
        torch.set_grad_enabled(False)
        encoder_model_dict = {}
        decoder_model_dict = {}

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
        if cfg.feed_forward_proj == "gated-gelu":
            activation_fn = "gelu_new"
        if not hasattr(cfg, "n_positions"):
            cfg.n_positions = 512
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

        shared = torch_model.shared.cuda()
        encoder_final_layernorm = torch_model.encoder.final_layer_norm.cuda()
        decoder_final_layernorm = torch_model.decoder.final_layer_norm.cuda()
        encoder_position_embedding = torch_model.encoder.block[0].layer[0].SelfAttention.relative_attention_bias
        decoder_position_embedding = torch_model.decoder.block[0].layer[0].SelfAttention.relative_attention_bias

        encoder = EETT5Encoder.from_torch(encoder_config, cfg, shared, encoder_position_embedding, encoder_final_layernorm, encoder_layer_model_dict, cfg.num_layers, data_type=data_type, bias=False)
        decoder = EETT5Decoder.from_torch(decoder_config, cfg, shared, decoder_position_embedding, decoder_final_layernorm, decoder_layer_model_dict, cfg.num_decoder_layers, data_type=data_type, bias=False)
        eet_model = EETT5Model(cfg, encoder, decoder)

        return eet_model       


class EETT5ForConditionalGeneration(GenerationMixin_EET):
    def __init__(self, t5model, lm_head, config):
        self.model = t5model
        self.lm_head = lm_head
        self.config = config
        self.main_input_name = "input_ids"
        self.device = "cuda:0"

    def get_encoder(self):
        return self.model.encoder

    def get_decoder(self):
        return self.model.decoder

    def _prepare_encoder_decoder_kwargs_for_generation(self, inputs_tensor, model_kwargs, model_input_name=None):
        # 1. get encoder
        encoder = self.get_encoder()

        # 2. prepare encoder args and encoder kwargs
        encoder_kwargs = {}
        attention_mask = model_kwargs["attention_mask"]

        if attention_mask is None:
            pre_padding_len = torch.empty(0).long()
        else:
            # transformers 0 - padding;1 - nopadding
            pre_padding_len = torch.sum(1 - attention_mask, 1).long().cuda()
        encoder_kwargs["normalize_before"] = True
        encoder_kwargs["pre_padding_len"] = pre_padding_len

        # 3. make sure that encoder returns `ModelOutput`
        model_input_name = model_input_name if model_input_name is not None else self.main_input_name
        encoder_kwargs[model_input_name] = inputs_tensor.cuda()
        
        encoder_last_hidden_state = encoder(**encoder_kwargs)
        encoder_outputs = BaseModelOutput(last_hidden_state=encoder_last_hidden_state)
        model_kwargs["encoder_outputs"]: BaseModelOutput = encoder_outputs

        return model_kwargs

    def prepare_inputs_for_generation(
        self,
        input_ids,
        first_pass=True, 
        past=None,
        attention_mask=None,
        head_mask=None,
        decoder_head_mask=None,
        cross_attn_head_mask=None,
        use_cache=None,
        encoder_outputs=None,
        pre_padding_len=None,
        **kwargs
    ):
        batch_size, seq_len, _ = encoder_outputs[0].size()
        # compute per sample length EET增量推理需要输入encoder out的真实句长
        encoder_seq_length = torch.tensor([seq_len] * batch_size).int().cuda()
        if attention_mask is not None:
            pre_padding_len = torch.sum(1 - attention_mask, 1).int().cuda()
            encoder_seq_length = encoder_seq_length - pre_padding_len

        # only last token for inputs_ids if past is defined in kwargs
        if first_pass == False:
            input_ids = input_ids[:, -1].unsqueeze(-1)

        attention_mask = kwargs.get("attention_mask", None)
        input_ids = input_ids.contiguous()
        encoder_outputs.last_hidden_state = encoder_outputs.last_hidden_state.contiguous()

        return {
            "decoder_input_ids": input_ids,
            "past_key_values": past,
            "attention_mask": attention_mask,
            "encoder_outputs": encoder_outputs.last_hidden_state,
            "encoder_seq_length": encoder_seq_length,
        }

    def forward(
        self,
        input_ids=None,
        attention_mask=None,
        head_mask=None,
        decoder_input_ids=None,
        decoder_attention_mask=None,
        encoder_outputs=None,
        past_key_values=None,
        labels=None,
        use_cache=None,
        encoder_seq_length=None,
        reorder_state=None,
        first_pass=True,
        self_past_key_values_length=0,
    ):
        
        transformer_outputs = self.model(
            input_ids=input_ids,
            encoder_outputs=encoder_outputs,
            encoder_seq_length=encoder_seq_length,
            first_pass=first_pass,
            attention_mask=attention_mask,
            decoder_input_ids=decoder_input_ids,
            decoder_attention_mask=decoder_attention_mask,
            reorder_state=reorder_state,
            self_past_key_values_length=self_past_key_values_length,
        )
        # if self.config.feed_forward_proj != "gated-gelu":
        #     transformer_outputs = transformer_outputs * (self.config.d_model**-0.5)
        lm_logits = self.lm_head(transformer_outputs)
        return Seq2SeqLMOutput(
            loss=None,
            logits=lm_logits,
            past_key_values=None,
            decoder_hidden_states=None,
            decoder_attentions=None,
            cross_attentions=None,
            encoder_last_hidden_state=None,
            encoder_hidden_states=None,
            encoder_attentions=None,
        )

    def __call__(
        self,
        input_ids=None,
        past_key_values=None,
        attention_mask=None,
        head_mask=None,
        encoder_outputs=None,
        decoder_input_ids=None,
        first_pass=True,
        encoder_seq_length=torch.empty(0),
        self_past_key_values_length=0,
        **kwargs,
    ):
        return self.forward(        
            input_ids=input_ids,
            past_key_values=past_key_values,
            attention_mask=attention_mask,
            head_mask=head_mask,
            encoder_outputs=encoder_outputs,
            decoder_input_ids=decoder_input_ids,
            first_pass=first_pass,
            encoder_seq_length=encoder_seq_length,
            self_past_key_values_length=self_past_key_values_length,
            )

    def from_pretrained(model_id_or_path: str, max_batch, data_type):
        """from_pretrained"""
        torch.set_grad_enabled(False)
        model_dict = {}
        embedding_dict = {}
        torch_model = T5ForConditionalGeneration.from_pretrained(model_id_or_path)
        if data_type == torch.float32:
            torch_model = torch_model.float()
        else:
            torch_model = torch_model.half()

        t5 = EETT5Model.from_pretrained(model_id_or_path, max_batch, data_type)

        lm_head = torch_model.lm_head.cuda()
        model = EETT5ForConditionalGeneration(t5, lm_head, torch_model.config)

        return model