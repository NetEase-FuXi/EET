#
# Created by zsd on 2022/04/20.
#
"""EET Bart model. """

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
from transformers import BartModel
from eet.transformers.encoder import *
from eet.utils.mapping import convert_name

from EET import MetaDesc as meta_desc
from EET import FeedForwardNetwork as eet_ffn
from EET import MultiHeadAttention as eet_attention
from EET import CrossMultiHeadAttention as eet_cross_attention
from EET import MaskedMultiHeadAttention as eet_masked_attention
from EET import Embedding as eet_embedding
from EET import LayerNorm as eet_layernorm

# different to other models, Bart automatically creates decoder_input_ids from
# input_ids if no decoder_input_ids are provided
def shift_tokens_right(input_ids: torch.Tensor, pad_token_id: int, decoder_start_token_id: int):
    """
    Shift input ids one token to the right.
    """
    shifted_input_ids = input_ids.new_zeros(input_ids.shape)
    shifted_input_ids[:, 1:] = input_ids[:, :-1].clone()
    shifted_input_ids[:, 0] = decoder_start_token_id

    if pad_token_id is None:
        raise ValueError("self.model.config.pad_token_id has to be defined.")
    # replace possible -100 values in labels by `pad_token_id`
    shifted_input_ids.masked_fill_(shifted_input_ids == -100, pad_token_id)

    return shifted_input_ids


class EETBartEmbedding():
    def __init__(self, config, embedding_dict, data_type=torch.float32):
        self.if_layernorm = True
        self.embedding_weights = embedding_dict['embeddings.word_embeddings.weight'].cuda().type(data_type)
        self.position_weights = embedding_dict['embeddings.position_embeddings.weight'].cuda().type(data_type)
        self.Layernorm_weights = embedding_dict['embeddings.layernorm.weight'].cuda().type(data_type)
        self.Layernorm_bias = embedding_dict['embeddings.layernorm.bias'].cuda().type(data_type)
        # not use token_type
        self.token_type_ids = torch.empty(0).long()
        self.token_type_weights = torch.empty(0)

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
        embedding = EETBartEmbedding(config, embedding_dict, data_type=data_type)
        return embedding


class EETBartEncoder():
    def __init__(self, cfg, EncoderLayers, embed_tokens, embed_positions, layernorm_embedding):
        self.layers = EncoderLayers
        self.embed_tokens = embed_tokens
        self.embed_positions = embed_positions
        self.layernorm_embedding = layernorm_embedding
        embed_dim = cfg.d_model
        self.embed_scale = math.sqrt(embed_dim) if cfg.scale_embedding else 1.0

    def __call__(
        self,
        input_ids,
        pre_padding_len=None,
        normalize_before=False,
        need_sequence_mask=False,
    ):
        input_shape = input_ids.size()
        input_ids = input_ids.view(-1, input_shape[-1])    
        inputs_embeds = self.embed_tokens(input_ids) * self.embed_scale        
        embed_pos = self.embed_positions(input_shape)
        hidden_states = inputs_embeds + embed_pos
        hidden_states = self.layernorm_embedding(hidden_states)
        for layer in self.layers:
            hidden_states = layer(
                hidden_states,
                pre_padding_len=pre_padding_len,
                normalize_before=normalize_before,
                need_sequence_mask=need_sequence_mask
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
        eet_encoder = EETBartEncoder(cfg, EncoderLayers, embed_tokens, embed_positions, layernorm_embedding)
        return eet_encoder


class EETBartDecoder():
    def __init__(self, cfg, DecoderLayers, embed_tokens, embed_positions, layernorm_embedding):
        self.layers = DecoderLayers
        self.embed_tokens = embed_tokens
        self.embed_positions = embed_positions
        self.layernorm_embedding = layernorm_embedding
        embed_dim = cfg.d_model
        self.embed_scale = math.sqrt(embed_dim) if cfg.scale_embedding else 1.0

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
        inputs_embeds = self.embed_tokens(input_ids) * self.embed_scale        
        embed_pos = self.embed_positions(input_shape)
        hidden_states = inputs_embeds + embed_pos
        hidden_states = self.layernorm_embedding(hidden_states)

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

        eet_decoder = EETBartDecoder(cfg, DecoderLayers, embed_tokens, embed_positions, layernorm_embedding)
        return eet_decoder


class EETBartModel():
    def __init__(self, cfg, encoder, decoder):
        self.encoder = encoder
        self.decoder = decoder
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
        if decoder_input_ids is None:
            decoder_input_ids = shift_tokens_right(
                input_ids, self.cfg.pad_token_id, self.cfg.decoder_start_token_id
            )            

        if attention_mask is None:
            pre_padding_len = self.pre_padding_len
        else:
            # transformers 0 - padding;1 - nopadding
            pre_padding_len = torch.sum(1 - attention_mask, 1).long().cuda()

        if encoder_out is None:
            encoder_out = copy.deepcopy(self.encoder(
                input_ids=input_ids,
                pre_padding_len=pre_padding_len,
                normalize_before=False,
                need_sequence_mask=False,
            ))
        if reorder_state is not None:
            self.reorder_state = reorder_state.long()

        decoder_out = self.decoder(
            input_ids=decoder_input_ids,
            encoder_out=encoder_out,
            first_pass=first_pass,
            pre_padding_len=pre_padding_len,
            encoder_attention_mask=self.encoder_attention_mask,
            head_mask=None,
            reorder_state=self.reorder_state,
            normalize_before=False,
        )

        return decoder_out


    @staticmethod
    def from_pretrained(model_id_or_path: str, max_batch, full_seq_len, data_type=torch.float32, device_id=0):
        """from torch."""
        torch.set_grad_enabled(False)
        encoder_model_dict = {}
        decoder_model_dict = {}

        torch_model = BartModel.from_pretrained(model_id_or_path)
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
        activation_fn = cfg.activation_function
        batch_size = max_batch
        encoder_config = meta_desc(batch_size, cfg.encoder_attention_heads, cfg.d_model, cfg.encoder_layers,
                                   cfg.max_position_embeddings, cfg.max_position_embeddings, data_type, device, False,
                                   activation_fn)
        decoder_config = meta_desc(batch_size, cfg.decoder_attention_heads, cfg.d_model, cfg.decoder_layers,
                                   cfg.max_position_embeddings, cfg.max_position_embeddings, data_type, device, False,
                                   activation_fn)

        if data_type==torch.float16:
            torch_model = torch_model.half()
        else:
            torch_model = torch_model.float()

        shared = torch_model.shared.cuda()
        encoder_embed_pos = torch_model.encoder.embed_positions.cuda()
        encoder_embed_layernorm = torch_model.encoder.layernorm_embedding.cuda()
        decoder_embed_pos = torch_model.decoder.embed_positions.cuda()
        decoder_embed_layernorm = torch_model.decoder.layernorm_embedding.cuda()

        encoder = EETBartEncoder.from_torch(encoder_config, cfg, encoder_layer_model_dict, cfg.encoder_layers, shared, encoder_embed_pos, encoder_embed_layernorm, data_type=data_type, bias=True)
        decoder = EETBartDecoder.from_torch(decoder_config, cfg, decoder_layer_model_dict, cfg.decoder_layers, shared, decoder_embed_pos, decoder_embed_layernorm, data_type=data_type, bias=True)
        eet_model = EETBartModel(cfg, encoder, decoder)

        return eet_model