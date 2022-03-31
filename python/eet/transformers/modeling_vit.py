#
# Created by zsd on 2022/03/01.
#
"""EET transformers vit model. """

import math
import time
import torch
import torch.nn as nn
import numpy as np
from torch import Tensor
from torch import functional as F
from torch.nn.parameter import Parameter
from typing import Any, Dict, List, Optional, Tuple
from transformers import ViTModel
from eet.transformers.modeling_transformer import *
from eet.utils.mapping import convert_name

from EET import MetaDesc as meta_desc
from EET import FeedForwardNetwork as eet_ffn
from EET import MultiHeadAttention as eet_attention
from EET import Embedding as eet_embedding
from EET import LayerNorm as eet_layernorm

Beginning_of_param = 8

__all__ = ['EETViTEmbedding', 'EETViTModel']

class EETViTEmbedding(nn.Module):
    def __init__(self, config, embedding_dict, data_type=torch.float32):
        super().__init__()
        self.patch_embeddings = nn.Conv2d(config.num_channels, config.hidden_size, kernel_size=config.patch_size, stride=config.patch_size)
        self.patch_embeddings.weight = Parameter(embedding_dict['embeddings.patch_embeddings.projection.weight'].cuda().type(data_type))
        self.patch_embeddings.bias = Parameter(embedding_dict['embeddings.patch_embeddings.projection.bias'].cuda().type(data_type))
        
        self.position_embeddings = Parameter(embedding_dict['embeddings.position_embeddings'].cuda().type(data_type))
        self.cls_token = Parameter(embedding_dict['embeddings.cls_token'].cuda().type(data_type))
        self.dropout = nn.Dropout(0.0)
        self.num_patches = (config.image_size // config.patch_size) * (config.image_size // config.patch_size)

    def __call__(self, pixel_values):
        batch_size, num_features, height, width = pixel_values.shape
        embeddings = self.patch_embeddings(pixel_values).flatten(2).transpose(1, 2)

        cls_tokens = self.cls_token.expand(batch_size, -1, -1)
        embeddings = torch.cat((cls_tokens, embeddings), dim=1)
        embeddings = embeddings + self.position_embeddings

        embeddings = self.dropout(embeddings)
        return embeddings

    @staticmethod
    def from_torch(config, embedding_dict, data_type=torch.float32):
        embedding = EETViTEmbedding(config, embedding_dict, data_type=data_type)
        return embedding


class EETViTPooler(nn.Module):
    def __init__(self, hidden_size, weight, bias, data_type=torch.float32):
        super().__init__()
        self.dense = nn.Linear(hidden_size, hidden_size)
        self.activation = nn.Tanh()
        self.dense.weight = Parameter(weight.cuda().type(data_type))
        self.dense.bias = Parameter(bias.cuda().type(data_type))
    
    def __call__(self, hidden_states):
        first_token_tensor = hidden_states[:, 0, :]
        pooled_output = self.dense(first_token_tensor)
        pooled_output = self.activation(pooled_output)
        return pooled_output
    
    @staticmethod
    def from_torch(hidden_size, weight, bias, data_type=torch.float32):
        pooler = EETViTPooler(hidden_size, weight, bias, data_type)
        return pooler


class EETViTModel():
    def __init__(self, config, embedding, encoder, layernorm, pooler):
        self.embedding = embedding
        self.encoder = encoder
        self.pre_padding_len = torch.empty(0).long()
        self.layernorm = layernorm
        self.pooler = pooler

    def __call__(
        self,
        pixel_values,
        attention_mask=None
    ):
        embedding_out = self.embedding(pixel_values)
        encoder_out = self.encoder(embedding_out, pre_padding_len=self.pre_padding_len, normalize_before=True)
        sequence_output = self.layernorm(encoder_out)
        pooled_output = self.pooler(sequence_output)
        
        return pooled_output

    @staticmethod
    def from_pretrained(model_id_or_path: str, max_batch, data_type):
        """from torch"""
        torch.set_grad_enabled(False)
        model_dict = {}
        embedding_dict = {}
        other_dict = {}
        torch_model = ViTModel.from_pretrained(model_id_or_path)
        model_name = type(torch_model).__name__
        cfg = torch_model.config

        # print ts model param dict
        for k, v in torch_model.state_dict().items():
            if 'embeddings.' in k:
                embedding_dict[k] = v
            elif 'layer.' in k:
                # Structure mapping
                k = convert_name(k, model_name)
                k = k[k.find('layer.'):]                 
                model_dict[k] = v
            else:
                other_dict[k] = v

        from itertools import groupby
        layer_model_dict = {k: dict(v) for k, v in groupby(list(model_dict.items()), 
                                                           lambda item: item[0][:(item[0].index('.', item[0].index('.')+1))])}

        device = "cuda:0"
        activation_fn = cfg.hidden_act
        batch_size = max_batch
        config = meta_desc(batch_size, cfg.num_attention_heads, cfg.hidden_size,
                           cfg.num_hidden_layers, cfg.hidden_size, cfg.hidden_size, data_type, device, False, activation_fn)

        embedding = EETViTEmbedding(cfg, embedding_dict, data_type)
        encoder = EETEncoder.from_torch(config, layer_model_dict, cfg.num_hidden_layers, data_type)
        layer_norm = EETLayerNorm.from_torch(config, other_dict['layernorm.weight'], other_dict['layernorm.bias'], data_type)
        pooler = EETViTPooler.from_torch(cfg.hidden_size, other_dict['pooler.dense.weight'], other_dict['pooler.dense.bias'], data_type)

        eet_model = EETViTModel(config, embedding, encoder, layer_norm, pooler)
        return eet_model
