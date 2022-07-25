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
from eet.transformers.encoder_decoder import *
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
    def __init__(self, config, embedding_dict, data_type=torch.float32, name='emb_cache'):
        self.if_layernorm = True
        self.embedding_weights = embedding_dict['embeddings.word_embeddings.weight'].cuda().type(data_type)
        self.position_weights = embedding_dict['embeddings.position_embeddings.weight'].cuda().type(data_type)
        self.Layernorm_weights = embedding_dict['embeddings.layernorm.weight'].cuda().type(data_type)
        self.Layernorm_bias = embedding_dict['embeddings.layernorm.bias'].cuda().type(data_type)
        # not use token_type
        self.token_type_ids = torch.empty(0).long()
        self.token_type_weights = torch.empty(0)

        self.embedding = eet_embedding(config, self.embedding_weights, self.position_weights,
                                       self.token_type_weights, self.Layernorm_weights, self.Layernorm_bias, name)

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
    def from_torch(config, embedding_dict, data_type=torch.float32, name='emb_cache'):
        embedding = EETBartEmbedding(config, embedding_dict, data_type=data_type, name=name)
        return embedding


class EETBartModel():
    def __init__(self, cfg, encoder_embedding, encoder, decoder_embedding, decoder):
        self.encoder_embedding = encoder_embedding
        self.decoder_embedding = decoder_embedding
        self.encoder = encoder
        self.decoder = decoder
        self.cfg = cfg
        self.offset = 2
        self.pre_padding_len = torch.empty(0).long()
        self.decoder_pre_padding_len = torch.empty(0).long()
        self.reorder_state = torch.empty(0).long()
        self.position_ids = torch.arange(0, cfg.max_position_embeddings).reshape(1, cfg.max_position_embeddings).cuda()

    def __call__(
        self,
        input_ids,
        encoder_outputs=None,
        encoder_seq_length=torch.empty(0),
        first_pass=True,
        attention_mask=None,
        decoder_input_ids=None,
        decoder_attention_mask=None,
        reorder_state=None,
        self_past_key_values_length=0,
    ):
        if decoder_input_ids is None:
            decoder_input_ids = shift_tokens_right(
                input_ids, self.cfg.pad_token_id, self.cfg.decoder_start_token_id
            )            
        decoder_input_shape = decoder_input_ids.size()

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
            assert per_sample_length is not None
            encoder_outputs = torch.empty(0)

        if encoder_outputs is None:
            input_shape = input_ids.size()
            input_ids = input_ids.view(-1, input_shape[-1])
            position_ids = self.position_ids[:, :input_shape[1]] + self.offset
            hidden_states = self.encoder_embedding(input_ids, position_ids, token_type_ids=None)
            encoder_outputs = self.encoder(
                hidden_states=hidden_states,
                pre_padding_len=pre_padding_len,
                normalize_before=False,
                need_sequence_mask=False,
            )
        if reorder_state is not None:
            self.reorder_state = reorder_state.long()

        position_ids = self.position_ids[:, self_past_key_values_length:self_past_key_values_length+decoder_input_shape[1]] + self.offset  
        hidden_states = self.decoder_embedding(decoder_input_ids, position_ids, token_type_ids=None)

        decoder_out = self.decoder(
            hidden_states=hidden_states,
            encoder_outputs=encoder_outputs,
            first_pass=first_pass,
            pre_padding_len=decoder_pre_padding_len,
            per_sample_length=per_sample_length,
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
        encoder_embedding_dict = {}
        decoder_model_dict = {}
        decoder_embedding_dict = {}

        torch_model = BartModel.from_pretrained(model_id_or_path)
        cfg = torch_model.config
        model_name = cfg.model_type

        for k, v in torch_model.state_dict().items():
            k = convert_name(k, model_name, verbose=False)
            if 'encoder.layer.' in k:
                k = k[k.find('layer.'):]
                encoder_model_dict[k] = v
            if 'encoder.embeddings.' in k:
                k = k[k.find('embeddings.'):]
                encoder_embedding_dict[k] = v
            if 'decoder.layer.' in k:
                k = k[k.find('layer.'):]
                decoder_model_dict[k] = v
            if 'decoder.embeddings.' in k:
                k = k[k.find('embeddings.'):]
                decoder_embedding_dict[k] = v

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

        encoder_embedding = EETBartEmbedding.from_torch(encoder_config, encoder_embedding_dict, data_type=data_type, name='encoder_out_cache')
        encoder = EETEncoder.from_torch(encoder_config, encoder_layer_model_dict, cfg.encoder_layers, data_type=data_type, bias=True)
        decoder_embedding = EETBartEmbedding.from_torch(decoder_config, decoder_embedding_dict, data_type=data_type, name='decoder_out_cache')
        decoder = EETDecoder.from_torch(decoder_config, decoder_layer_model_dict, cfg.decoder_layers, data_type=data_type, bias=True)
        eet_model = EETBartModel(cfg, encoder_embedding, encoder, decoder_embedding, decoder)
        
        return eet_model
