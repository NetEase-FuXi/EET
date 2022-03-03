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

from EET import MetaDesc as meta_desc
from EET import FeedForwardNetwork as eet_ffn
from EET import MultiHeadAttention as eet_attention
from EET import Embedding as eet_embedding
from EET import LayerNorm as eet_layernorm

Beginning_of_param = 8

__all__ = [
    'EETViTEmbedding', 'EETViTFeedforward', 'EETViTAttention', 'EETViTEncoderLayer', 'EETViTEncoder', 'EETViTModel'
]

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


class EETLayerNorm():
    def __init__(self, config, layernorm_weights, layernorm_bias, data_type=torch.float32):
        self.layernorm_weights = layernorm_weights.cuda().type(data_type)
        self.layernorm_bias = layernorm_bias.cuda().type(data_type)
        self.layernorm = eet_layernorm(config, self.layernorm_weights, self.layernorm_bias)

    def __call__(self,
                 x):
        return self.layernorm.layer_norm(x)
    
    @staticmethod
    def from_torch(config, layernorm_weights, layernorm_bias,data_type = torch.float32):
        layernorm = EETLayerNorm(config, layernorm_weights, layernorm_bias, data_type=data_type)
        return layernorm


class EETViTPooler(nn.Module):
    def __init__(self, hidden_size, model_dict, data_type=torch.float32):
        super().__init__()
        self.dense = nn.Linear(hidden_size, hidden_size)
        self.activation = nn.Tanh()
        self.dense.weight = Parameter(model_dict['pooler.dense.weight'].cuda().type(data_type))
        self.dense.bias = Parameter(model_dict['pooler.dense.bias'].cuda().type(data_type))
    
    def __call__(self, hidden_states):
        first_token_tensor = hidden_states[:, 0, :]
        pooled_output = self.dense(first_token_tensor)
        pooled_output = self.activation(pooled_output)
        return pooled_output
    
    @staticmethod
    def from_torch(hidden_size, model_dict, data_type=torch.float32):
        pooler = EETViTPooler(hidden_size, model_dict, data_type)
        return pooler
    

class EETViTFeedforward():
    def __init__(self, config, model_dict, layer_id, data_type=torch.float32):
        self.intermediate_weights = torch.t([x[1] for x in model_dict.items() if 'intermediate.dense.weight' in x[0]][0]).contiguous().cuda().type(data_type)
        self.intermediate_bias = [x[1] for x in model_dict.items() if 'intermediate.dense.bias' in x[0]][0].cuda().type(data_type)
        self.output_weights = torch.t([x[1] for x in model_dict.items() if str(layer_id)+'.output.dense.weight' in x[0]][0]).contiguous().cuda().type(data_type)
        self.output_bias = [x[1] for x in model_dict.items() if str(layer_id)+'.output.dense.bias' in x[0]][0].cuda().type(data_type)
        self.layernorm_weights = [x[1] for x in model_dict.items() if str(layer_id)+'.layernorm_after.weight' in x[0]][0].cuda().type(data_type)
        self.layernorm_bias = [x[1] for x in model_dict.items() if str(layer_id)+'.layernorm_after.bias' in x[0]][0].cuda().type(data_type)

        self.ffn = eet_ffn(config, self.intermediate_weights, self.intermediate_bias, self.output_weights, self.output_bias, self.layernorm_weights, self.layernorm_bias)

    def __call__(self,
                 x,
                 pre_layernorm=True,
                 add_redusial=True):
        return self.ffn.forward(x, pre_layernorm, add_redusial)

    @staticmethod
    def from_torch(config, model_dict, layer_id, data_type=torch.float32):
        feedforward = EETViTFeedforward(config, model_dict, layer_id, data_type=data_type)
        return feedforward


class EETViTAttention():
    def __init__(self, config, model_dict, layer_id, data_type=torch.float32):
        q_weights = [x[1] for x in model_dict.items() if 'attention.query.weight' in x[0]][0].contiguous().cuda().type(data_type)
        k_weights = [x[1] for x in model_dict.items() if 'attention.key.weight' in x[0]][0].contiguous().cuda().type(data_type)
        v_weights = [x[1] for x in model_dict.items() if 'attention.value.weight' in x[0]][0].contiguous().cuda().type(data_type)
        self.qkv_weight = torch.cat((q_weights, k_weights, v_weights), 0).transpose(0, 1).contiguous()
        self.q_bias = [x[1] for x in model_dict.items() if 'attention.query.bias' in x[0]][0].cuda().type(data_type)
        self.k_bias = [x[1] for x in model_dict.items() if 'attention.key.bias' in x[0]][0].cuda().type(data_type)
        self.v_bias = [x[1] for x in model_dict.items() if 'attention.value.bias' in x[0]][0].cuda().type(data_type)
        self.out_weights = torch.t([x[1] for x in model_dict.items() if 'attention.output.dense.weight' in x[0]][0]).contiguous().cuda().type(data_type)
        self.out_bias = [x[1] for x in model_dict.items() if 'attention.output.dense.bias' in x[0]][0].cuda().type(data_type)
        self.layernorm_weights = [x[1] for x in model_dict.items() if 'layernorm_before.weight' in x[0]][0].cuda().type(data_type)
        self.layernorm_bias = [x[1] for x in model_dict.items() if 'layernorm_before.bias' in x[0]][0].cuda().type(data_type)

        self.attention = eet_attention(config, self.qkv_weight, self.q_bias, self.k_bias, self.v_bias,
                                       self.out_weights, self.out_bias, self.layernorm_weights, self.layernorm_bias)

    def __call__(self,
                 x,
                 pre_padding_len,
                 pre_layernorm=False,
                 add_redusial=True):
        return self.attention.forward(x, pre_padding_len, pre_layernorm, add_redusial)

    @staticmethod
    def from_torch(config, model_dict, layer_id, data_type=torch.float32):
        attention = EETViTAttention(config, model_dict, layer_id, data_type=data_type)
        return attention


class EETViTEncoderLayer():
    def __init__(self, config, attention, feedforward):
        self.attention = attention
        self.feedfoward = feedforward

    def __call__(self,
                 x,
                 pre_padding_len,
                 normalize_before=True):
        self_attn_out = self.attention(x=x,
                                       pre_padding_len=pre_padding_len,
                                       pre_layernorm=normalize_before,
                                       add_redusial=True)
        out = self.feedfoward(self_attn_out,
                              pre_layernorm=normalize_before,
                              add_redusial=True)
        return out

    @staticmethod
    def from_torch(config, model_dict, layer_id, data_type=torch.float32):
        attention = EETViTAttention.from_torch(
            config=config, model_dict=model_dict, layer_id=layer_id, data_type=data_type)
        feedforward = EETViTFeedforward.from_torch(
            config=config, model_dict=model_dict, layer_id=layer_id, data_type=data_type)
        layer = EETViTEncoderLayer(config, attention, feedforward)
        return layer



class EETViTEncoder():
    def __init__(self, EncoderLayers):
        self.layers = EncoderLayers

    def __call__(
        self,
        x,
        pre_padding_len=None,
        normalize_before=True
    ):
        for layer in self.layers:
            x = layer(x,
                      pre_padding_len=pre_padding_len,
                      normalize_before=True)
        return x

    @staticmethod
    def from_torch(config, layer_model_dict, layer_num, data_type=torch.float32):
        """from torch."""
        EncoderLayers = []
        for i in range(layer_num):
            if i < 10:
                EncoderLayers.extend(
                    [
                        EETViTEncoderLayer.from_torch(
                            config, layer_model_dict['layer.'+str(i)+'.'], i, data_type=data_type)
                    ]
                )
            else:
                EncoderLayers.extend(
                    [
                        EETViTEncoderLayer.from_torch(
                            config, layer_model_dict['layer.'+str(i)], i, data_type=data_type)
                    ]
                )

        eet_encoder = EETViTEncoder(EncoderLayers)
        return eet_encoder


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
        layernorm_dict = {}
        pooler_dict = {}
        torch_model = ViTModel.from_pretrained(model_id_or_path)
        cfg = torch_model.config

        # print ts model param dict
        for k, v in torch_model.state_dict().items():
            if 'embeddings.' in k:
                embedding_dict[k] = v
            elif 'layer.' in k:
                k = k[Beginning_of_param:]
                model_dict[k] = v
            elif 'layernorm.' in k:
                layernorm_dict[k] = v
            else:
                pooler_dict[k] = v

        from itertools import groupby
        layer_model_dict = {k: dict(v) for k, v in groupby(
            list(model_dict.items()), lambda item: item[0][:Beginning_of_param])}
        
        device = "cuda:0"
        activation_fn = cfg.hidden_act
        batch_size = max_batch
        config = meta_desc(batch_size, cfg.num_attention_heads, cfg.hidden_size,
                           cfg.num_hidden_layers, cfg.hidden_size, cfg.hidden_size, data_type, device, False, activation_fn)

        embedding = EETViTEmbedding(cfg, embedding_dict, data_type)
        encoder = EETViTEncoder.from_torch(config, layer_model_dict, cfg.num_hidden_layers, data_type)
        layer_norm = EETLayerNorm.from_torch(config, layernorm_dict['layernorm.weight'], layernorm_dict['layernorm.bias'], data_type)
        pooler = EETViTPooler.from_torch(cfg.hidden_size, pooler_dict, data_type)

        eet_model = EETViTModel(config, embedding, encoder, layer_norm, pooler)
        return eet_model
