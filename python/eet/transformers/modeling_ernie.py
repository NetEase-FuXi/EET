#
# Created by zsd on 2023/05/19.
#
"""EET transformers ernie model. """

import math
import time
import torch
import torch.nn as nn
import numpy as np
from torch import Tensor
from typing import Any, Dict, List, Optional, Tuple
from transformers import (
    ErnieModel,
)
from transformers.file_utils import ModelOutput
from transformers.modeling_outputs import (
    MaskedLMOutput,
)
from eet.transformers.encoder_decoder import *
from eet.utils.mapping import convert_name

from EET import MetaDesc as meta_desc
from EET import FeedForwardNetwork as eet_ffn
from EET import MultiHeadAttention as eet_attention
from EET import Embedding as eet_embedding

__all__ = ['EETErnieEmbedding', 'EETErnieModel']


class EETErnieEmbedding():
    def __init__(self, config, embedding_dict, data_type=torch.float32, name='emb_cache'):
        self.if_layernorm = True
        self.embedding_weights = embedding_dict['embeddings.word_embeddings.weight'].cuda().type(data_type)
        self.position_weights = embedding_dict['embeddings.position_embeddings.weight'].cuda().type(data_type)
        self.token_type_weights = embedding_dict['embeddings.token_type_embeddings.weight'].cuda().type(data_type)
        self.Layernorm_weights = embedding_dict['embeddings.LayerNorm.weight'].cuda().type(data_type)
        self.Layernorm_bias = embedding_dict['embeddings.LayerNorm.bias'].cuda().type(data_type)
        self.embedding = eet_embedding(config, self.embedding_weights, self.position_weights, self.token_type_weights, self.Layernorm_weights, self.Layernorm_bias, name)

    def __call__(self,
                 input_ids,
                 position_ids,
                 token_type_ids):
        return self.embedding.forward_transformers(input_ids, position_ids, token_type_ids, self.if_layernorm)

    @staticmethod
    def from_torch(config, embedding_dict, data_type=torch.float32, name='emb_cache'):
        embedding = EETErnieEmbedding(config, embedding_dict, data_type=data_type, name=name)
        return embedding


class EETErnieModel():
    def __init__(self, config, embedding, encoder, pooler=None):
        self.embedding = embedding
        self.encoder = encoder
        self.pooler = pooler
        self.pre_padding_len = torch.empty(0).long()
        self.position_embedding_type = getattr(config, "position_embedding_type", "absolute")
        self.position_ids = torch.arange(0, config.max_position_embeddings).reshape(1, config.max_position_embeddings).long().cuda()
        self.use_task_id = config.use_task_id

    def __call__(
        self,
        input_ids,
        position_ids=None,
        token_type_ids=None,
        task_type_ids=None,
        attention_mask=None,
    ):
        '''
        attention_mask:attention_padding_mask(:obj:`torch.FloatTensor` of shape :obj:`(batch_size, sequence_length)`, `optional`, defaults to :obj:`None`):
            Mask to avoid performing attention on the padding token indices of the encoder input.)
            Mask values selected in ``[0, 1]``:
            ``1`` for tokens that are NOT MASKED, ``0`` for MASKED tokens.
        '''
        input_shape = input_ids.size()
        if attention_mask is None:
            pre_padding_len = self.pre_padding_len
        else:
            # transformers 0 - padding;1 - nopadding
            pre_padding_len = torch.sum(1 - attention_mask, 1).long().cuda()

        embedding_out = self.embedding(input_ids, token_type_ids, task_type_ids, position_ids)

        sequence_output = self.encoder(embedding_out,
                                       pre_padding_len=pre_padding_len,
                                       normalize_before=False)
        pooled_output = self.pooler(sequence_output) if self.pooler is not None else None
        return sequence_output, pooled_output

    @staticmethod
    def from_pretrained(model_id_or_path: str, max_batch, max_seq_len=512, data_type=torch.float32, device_id=0):
        """from_pretrained."""
        torch.set_grad_enabled(False)
        model_dict = {}
        embedding_dict = {}
        torch_model = ErnieModel.from_pretrained(model_id_or_path)
        cfg = torch_model.config
        model_name = 'ernie'  # cfg.model_type

        for k, v in torch_model.state_dict().items():
            if 'embeddings.' in k:
                embedding_dict[k] = v
            if 'layer.' in k:
                # Structure mapping
                k = convert_name(k, model_name)
                k = k[k.find('layer.'):]
                model_dict[k] = v

        # Group by layer id in model_dict's keys
        from itertools import groupby
        layer_model_dict = {k: dict(v) for k, v in groupby(list(model_dict.items()),
                                                           lambda item: item[0][:(item[0].index('.', item[0].index('.')+1))])}

        device = "cpu" if device_id < 0 else f"cuda:{device_id}"
        activation_fn = cfg.hidden_act
        batch_size = max_batch
        config = meta_desc(dtype=data_type,
                           batch_size=batch_size,
                           head_num=cfg.num_attention_heads,
                           hidden_units=cfg.hidden_size,
                           layer_num=cfg.num_hidden_layers,
                           max_seq_len=max_seq_len,
                           activation_fn=activation_fn,
                           cuda_device=device)

        # embedding = EETErnieEmbedding.from_torch(config, embedding_dict, data_type)
        encoder = EETEncoder.from_torch(config, layer_model_dict, cfg.num_hidden_layers, data_type)
        if data_type == torch.float32:
            torch_model = torch_model.float()
        else:
            torch_model = torch_model.half()
        embedding = torch_model.embeddings.cuda()
        pooler = torch_model.pooler.cuda()
        eet_model = EETErnieModel(cfg, embedding, encoder, pooler)
        return eet_model

    def from_torch(torch_model, max_batch, max_seq_len=512, data_type=torch.float32, device_id=0, model_attr="ernie"):
        """from torch."""
        torch.set_grad_enabled(False)
        model_dict = {}
        embedding_dict = {}
        cfg = torch_model.config
        model_name = 'ernie'  # cfg.model_type

        for k, v in getattr(torch_model, model_attr).state_dict().items():
            if 'embeddings.' in k:
                embedding_dict[k] = v
            if 'layer.' in k:
                # Structure mapping
                k = convert_name(k, model_name)
                k = k[k.find('layer.'):]
                model_dict[k] = v

        # group by 'layer.n'
        from itertools import groupby
        layer_model_dict = {k: dict(v) for k, v in groupby(list(model_dict.items()),
                                                           lambda item: item[0][:(item[0].index('.', item[0].index('.')+1))])}

        device = "cpu" if device_id < 0 else f"cuda:{device_id}"
        activation_fn = cfg.hidden_act
        batch_size = max_batch
        config = meta_desc(dtype=data_type,
                           batch_size=batch_size,
                           head_num=cfg.num_attention_heads,
                           hidden_units=cfg.hidden_size,
                           layer_num=cfg.num_hidden_layers,
                           max_seq_len=max_seq_len,
                           activation_fn=activation_fn,
                           cuda_device=device)

        # embedding = EETErnieEmbedding.from_torch(config, embedding_dict, data_type)
        encoder = EETEncoder.from_torch(config, layer_model_dict, cfg.num_hidden_layers, data_type)
        if data_type == torch.float32:
            torch_model = torch_model.float()
        else:
            torch_model = torch_model.half()
        embedding = torch_model.ernie.embeddings.cuda()
        pooler = torch_model.ernie.pooler.cuda()
        eet_model = EETErnieModel(cfg, embedding, encoder, pooler)
        return eet_model
