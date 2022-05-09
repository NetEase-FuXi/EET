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
from eet.transformers.encoder import *
from eet.utils.mapping import convert_name

from EET import MetaDesc as meta_desc
from EET import FeedForwardNetwork as eet_ffn
from EET import MultiHeadAttention as eet_attention
from EET import CrossMultiHeadAttention as eet_cross_attention
from EET import MaskedMultiHeadAttention as eet_masked_attention
from EET import Embedding as eet_embedding
from EET import LayerNorm as eet_layernorm

__all__ = ['EETT5Embedding', 'EETT5Model']


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
                                       self.token_type_weights, self.Layernorm_weights, self.Layernorm_bias)

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


class EETT5Encoder():
    def __init__(self, cfg, EncoderLayers, embed_tokens):
        self.layers = EncoderLayers
        self.embed_tokens = embed_tokens

    def __call__(
        self,
        input_ids,
        pre_padding_len=None,
        normalize_before=False,
        need_sequence_mask=False,
    ):
        for layer in self.layers:
            hidden_states = layer(
                hidden_states,
                pre_padding_len=pre_padding_len,
                normalize_before=normalize_before,
                need_sequence_mask=need_sequence_mask,
            )
        return hidden_states

    @staticmethod
    def from_torch(config, cfg, layer_model_dict, layer_num, embed_tokens, embed_positions, layernorm_embedding, data_type=torch.float32, bias=True):
        """from torch."""
        EncoderLayers = []
        for i in range(layer_num):
            EncoderLayers.extend(
                [
                    EETEncoderLayer.from_torch(config, layer_model_dict['layer.' + str(i)], i, data_type=data_type, bias=bias)
                ]
            )
        eet_encoder = EETT5Encoder(cfg, EncoderLayers, embed_tokens, embed_positions, layernorm_embedding)
        return eet_encoder


class EETT5Decoder():
    def __init__(self, cfg, DecoderLayers, embed_tokens, embed_positions, layernorm_embedding):
        self.layers = DecoderLayers
        self.embed_tokens = embed_tokens

    def __call__(
        self,
        input_ids,
        encoder_out=None,
        first_pass=True,
        pre_padding_len=None,
        encoder_attention_mask=None,
        head_mask=None,
        reorder_state=None,
        normalize_before=False,
    ):
        input_shape = input_ids.size()
        input_ids = input_ids.view(-1, input_shape[-1])
        inputs_embeds = self.embed_tokens(input_ids)
        hidden_states = inputs_embeds

        # for layer in self.layers:
        for idx, layer in enumerate(self.layers):
            hidden_states = layer(
                hidden_states,
                encoder_out=encoder_out,
                first_pass=first_pass,
                pre_padding_len=pre_padding_len,
                encoder_attention_mask=encoder_attention_mask,
                head_mask=None,
                reorder_state=reorder_state,
                normalize_before=normalize_before,
                add_redusial=True,
            )

        return hidden_states

    @staticmethod
    def from_torch(config, cfg, layer_model_dict, layer_num, embed_tokens, embed_positions, layernorm_embedding, data_type=torch.float32, bias=True):
        """from torch."""
        DecoderLayers = []
        for i in range(layer_num):
            DecoderLayers.extend(
                [
                    EETDecoderLayer.from_torch(config, layer_model_dict['layer.' + str(i)], i, data_type=data_type, add_cross_attn=True, bias=bias, is_standard=True)
                ]
            )

        eet_decoder = EETT5Decoder(cfg, DecoderLayers, embed_tokens, embed_positions, layernorm_embedding)
        return eet_decoder


class EETT5Model():
    def __init__(self, cfg, shared, encoder, encoder_final_layernorm, decoder, decoder_final_layernorm):
        self.shared = shared
        self.encoder = encoder
        self.decoder = decoder
        self.encoder_final_layernorm = encoder_final_layernorm
        self.decoder_final_layernorm = decoder_final_layernorm
        self.cfg = cfg
        self.pre_padding_len = torch.empty(0).long()
        self.reorder_state = torch.empty(0).long()
        self.encoder_attention_mask = torch.empty(0)

    def __call__(
        self,
        input_ids,
        encoder_out=None,
        first_pass=True,
        attention_mask=None,
        decoder_input_ids=None,
        reorder_state=None,
    ):        
        if attention_mask is None:
            pre_padding_len = self.pre_padding_len
        else:
            # transformers 0 - padding;1 - nopadding
            pre_padding_len = torch.sum(1 - attention_mask, 1).long().cuda()

        input_shape = input_ids.size()
        input_ids = input_ids.view(-1, input_shape[-1])
        inputs_embeds = self.shared(input_ids)
        print('eet input embeds: ', inputs_embeds)
        if encoder_out is None:
            encoder_out = copy.deepcopy(self.encoder(
                hidden_states=inputs_embeds,
                pre_padding_len=pre_padding_len,
                normalize_before=False,
                need_sequence_mask=False,
            ))
            print('eet encoder out: ', encoder_out)
            encoder_out = self.encoder_final_layernorm(encoder_out)
        if reorder_state is not None:
            self.reorder_state = reorder_state.long()

        decoder_inputs_embeds = self.shared(decoder_input_ids)

        decoder_out = self.decoder(
            hidden_states=decoder_inputs_embeds,
            encoder_out=encoder_out,
            first_pass=first_pass,
            pre_padding_len=pre_padding_len,
            encoder_attention_mask=self.encoder_attention_mask,
            head_mask=None,
            reorder_state=self.reorder_state,
            normalize_before=False,
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
            k = convert_name(k, model_name, verbose=True)
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

        shared = torch_model.shared.cuda()
        encoder_final_layernorm = torch_model.encoder.final_layer_norm.cuda()
        decoder_final_layernorm = torch_model.decoder.final_layer_norm.cuda()

        encoder = EETEncoder.from_torch(encoder_config, encoder_layer_model_dict, cfg.num_layers, data_type=data_type, bias=False)
        decoder = EETDecoder.from_torch(decoder_config, decoder_layer_model_dict, cfg.num_decoder_layers, data_type=data_type, bias=False)
        eet_model = EETT5Model(cfg, shared, encoder, encoder_final_layernorm, decoder, decoder_final_layernorm)

        return eet_model