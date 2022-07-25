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
from transformers import ViTModel, ViTForMaskedImageModeling, ViTForImageClassification
from transformers.modeling_outputs import MaskedLMOutput, SequenceClassifierOutput
from eet.transformers.encoder_decoder import *
from eet.utils.mapping import convert_name

from EET import MetaDesc as meta_desc
from EET import FeedForwardNetwork as eet_ffn
from EET import MultiHeadAttention as eet_attention
from EET import Embedding as eet_embedding
from EET import LayerNorm as eet_layernorm

__all__ = ['EETViTEmbedding', 'EETViTModel','EETViTForMaskedImageModeling','EETViTForImageClassification']

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
        self.data_type = data_type
    def __call__(self, pixel_values):
        batch_size, num_features, height, width = pixel_values.shape
        pixel_values = pixel_values.to(self.data_type)
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
        pooled_output = self.pooler(sequence_output) if self.pooler is not None else None
        
        return sequence_output, pooled_output

    @staticmethod
    def from_pretrained(model_id_or_path: str, max_batch, data_type, device_id=0):
        """from_pretrained"""
        torch.set_grad_enabled(False)
        model_dict = {}
        embedding_dict = {}
        other_dict = {}
        torch_model = ViTModel.from_pretrained(model_id_or_path)
        cfg = torch_model.config
        model_name = cfg.model_type

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

        # Group by layer id in model_dict's keys
        from itertools import groupby
        layer_model_dict = {k: dict(v) for k, v in groupby(list(model_dict.items()), 
                                                           lambda item: item[0][:(item[0].index('.', item[0].index('.')+1))])}

        device = "cpu" if device_id < 0 else f"cuda:{device_id}"
        activation_fn = cfg.hidden_act
        batch_size = max_batch
        config = meta_desc(batch_size, cfg.num_attention_heads, cfg.hidden_size,
                           cfg.num_hidden_layers, cfg.hidden_size, cfg.hidden_size, data_type, device, False, activation_fn)

        embedding = EETViTEmbedding(cfg, embedding_dict, data_type)
        encoder = EETEncoder.from_torch(config, layer_model_dict, cfg.num_hidden_layers, data_type)
        layer_norm = EETLayerNorm.from_torch(config, other_dict['layernorm.weight'], other_dict['layernorm.bias'], data_type)
        if data_type == torch.float32:
            torch_model = torch_model.float()
        else:
            torch_model = torch_model.half()
        eet_model = EETViTModel(config, embedding, encoder, layer_norm, torch_model.pooler.to(device))
        return eet_model

    def from_torch(torch_model, max_batch, data_type, device_id=0):
        """from torch"""
        torch.set_grad_enabled(False)
        model_dict = {}
        embedding_dict = {}
        other_dict = {}

        cfg = torch_model.config
        model_name = cfg.model_type

        # print ts model param dict
        for k, v in torch_model.vit.state_dict().items():
            if 'embeddings.' in k:
                embedding_dict[k] = v
            elif 'layer.' in k:
                # Structure mapping
                k = convert_name(k, model_name)
                k = k[k.find('layer.'):]                 
                model_dict[k] = v
            else:
                other_dict[k] = v

        # Group by layer id in model_dict's keys
        from itertools import groupby
        layer_model_dict = {k: dict(v) for k, v in groupby(list(model_dict.items()), 
                                                           lambda item: item[0][:(item[0].index('.', item[0].index('.')+1))])}

        device = "cpu" if device_id < 0 else f"cuda:{device_id}"
        activation_fn = cfg.hidden_act
        batch_size = max_batch
        config = meta_desc(batch_size, cfg.num_attention_heads, cfg.hidden_size,
                           cfg.num_hidden_layers, cfg.hidden_size, cfg.hidden_size, data_type, device, False, activation_fn)

        embedding = EETViTEmbedding(cfg, embedding_dict, data_type)
        encoder = EETEncoder.from_torch(config, layer_model_dict, cfg.num_hidden_layers, data_type)
        layer_norm = EETLayerNorm.from_torch(config, other_dict['layernorm.weight'], other_dict['layernorm.bias'], data_type)

        if data_type == torch.float32:
            torch_model = torch_model.float()
        else:
            torch_model = torch_model.half()

        eet_model = EETViTModel(config, embedding, encoder, layer_norm, torch_model.pooler.to(device))
        return eet_model

class EETViTForMaskedImageModeling():
    def __init__(self,vit,decoder,config):
        self.config = config
        self.vit = vit
        self.decoder = decoder

    def __call__(
        self,
        pixel_values,
        attention_mask = None,
    ) :
        num_choices = pixel_values.shape[1]
        sequence_output, pooled_output = self.vit(
            pixel_values = pixel_values,
            attention_mask=attention_mask,
        )
        sequence_output = sequence_output[:, 1:]
        batch_size, sequence_length, num_channels = sequence_output.shape
        height = width = int(sequence_length**0.5)
        sequence_output = sequence_output.permute(0, 2, 1).reshape(batch_size, num_channels, height, width)

        reconstructed_pixel_values = self.decoder(sequence_output)

        return MaskedLMOutput(
            loss=None,
            logits=reconstructed_pixel_values,
            hidden_states=None,
            attentions=None,
        )

    def from_pretrained(model_id_or_path: str,max_batch, data_type):
        """from_pretrained"""
        torch.set_grad_enabled(False)
        model_dict = {}
        embedding_dict = {}
        torch_model = ViTForMaskedImageModeling.from_pretrained(model_id_or_path)
        vit = EETViTModel.from_torch(torch_model,max_batch,data_type)
        decoder = torch_model.decoder.cuda()
        model =  EETViTForMaskedImageModeling(vit, decoder,torch_model.config)

        return model

class EETViTForImageClassification():
    def __init__(self,vit,classifier,config):
        self.config = config
        self.vit = vit
        self.classifier = classifier

    def __call__(
        self,
        pixel_values,
        attention_mask = None,
    ) :
        sequence_output, pooled_output = self.vit(
            pixel_values = pixel_values,
            attention_mask=attention_mask,
        )

        logits = self.classifier(sequence_output[:, 0, :])

        return SequenceClassifierOutput(
            loss=None,
            logits=logits,
            hidden_states=None,
            attentions=None,
        )

    def from_pretrained(model_id_or_path: str,max_batch, data_type):
        """from_pretrained"""
        torch.set_grad_enabled(False)
        model_dict = {}
        embedding_dict = {}
        torch_model = ViTForImageClassification.from_pretrained(model_id_or_path)
        vit = EETViTModel.from_torch(torch_model,max_batch,data_type)
        classifier = torch_model.classifier.cuda()
        model =  EETViTForImageClassification(vit, classifier,torch_model.config)
        return model
