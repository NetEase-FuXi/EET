#
# Created by zsd on 2022/03/10.
#
"""EET transformers clip model. """

import math
import time
import torch
import torch.nn as nn
import numpy as np
from torch import Tensor
from torch import functional as F
from torch.nn.parameter import Parameter
from typing import Any, Dict, List, Optional, Tuple
from transformers import CLIPModel, ViTModel
from eet.transformers.encoder_decoder import *
from eet.utils.mapping import convert_name

from EET import MetaDesc as meta_desc
from EET import FeedForwardNetwork as eet_ffn
from EET import MultiHeadAttention as eet_attention
from EET import Embedding as eet_embedding
from EET import LayerNorm as eet_layernorm


class EETCLIPVisionEmbedding():
    def __init__(self, config, embedding_dict, data_type=torch.float32):
        self.config = config
        self.embed_dim = config.hidden_size
        self.image_size = config.image_size
        self.patch_size = config.patch_size
        self.num_positions = (self.image_size // self.patch_size) ** 2 + 1
        
        self.patch_embeddings = nn.Conv2d(
            config.num_channels, self.embed_dim,
            kernel_size=self.patch_size, stride=self.patch_size, bias=False
        )
        self.position_embeddings = nn.Embedding(self.num_positions, self.embed_dim)
        self.data_type = data_type
        self.cls_token = Parameter(embedding_dict['embeddings.class_embedding'].cuda().type(data_type))
        self.patch_embeddings.weight = Parameter(embedding_dict['embeddings.patch_embedding.weight'].cuda().type(data_type))
        self.position_ids = torch.arange(self.num_positions).expand((1, -1)).cuda()
        self.position_embeddings.weight = Parameter(embedding_dict['embeddings.position_embedding.weight'].cuda().type(data_type))
        self.dropout = nn.Dropout(config.dropout)

    def __call__(self, pixel_values, position_ids=None):
        if position_ids is None:
            position_ids = self.position_ids
        batch_size = pixel_values.shape[0]
        pixel_values = pixel_values.to(self.data_type)
        embeddings = self.patch_embeddings(pixel_values).flatten(2).transpose(1, 2)

        cls_tokens = self.cls_token.expand(batch_size, 1, -1)
        embeddings = torch.cat((cls_tokens, embeddings), dim=1)
        embeddings = embeddings + self.position_embeddings(position_ids)
        embeddings = self.dropout(embeddings)
        return embeddings

    @staticmethod
    def from_torch(config, embedding_dict, data_type=torch.float32):
        embedding = EETCLIPVisionEmbedding(config, embedding_dict, data_type=data_type)
        return embedding


class EETCLIPVisionTransformer():
    def __init__(self, config, embedding, encoder, pre_layernorm, post_layernorm):
        self.embedding = embedding
        self.pre_layernorm = pre_layernorm
        self.encoder = encoder
        self.post_layernorm = post_layernorm
        self.pre_padding_len = torch.empty(0).long()
    
    def __call__(self, pixel_values, position_ids=None):
        embedding_output = self.embedding(pixel_values, position_ids=position_ids)
        hidden_states = self.pre_layernorm(embedding_output)

        encoder_output = self.encoder(
            hidden_states,
            pre_padding_len=self.pre_padding_len,
            normalize_before=True
        )
        vision_output = encoder_output[:, 0, :]
        vision_output = self.post_layernorm(vision_output)

        return vision_output
    
    @staticmethod
    def from_torch(config, cfg, model_dict, data_type=torch.float32):
        pre_layernorm = EETLayerNorm.from_torch(config, model_dict['layernorm']['vision_model.pre_layrnorm.weight'], model_dict['layernorm']['vision_model.pre_layrnorm.bias'], data_type)
        embedding = EETCLIPVisionEmbedding.from_torch(cfg, model_dict['embeddings'], data_type)
        encoder = EETEncoder.from_torch(config, model_dict, cfg.num_hidden_layers, data_type)
        post_layernorm = EETLayerNorm.from_torch(config, model_dict['layernorm']['vision_model.post_layernorm.weight'], model_dict['layernorm']['vision_model.post_layernorm.bias'], data_type)

        vision_model = EETCLIPVisionTransformer(cfg, embedding, encoder, pre_layernorm, post_layernorm)
        return vision_model


class EETCLIPTextEmbedding():
    def __init__(self, config, embedding_dict, data_type=torch.float32):
        self.if_layernorm = False
        self.embedding_weights = embedding_dict['embeddings.token_embedding.weight'].cuda().type(data_type)
        self.position_weights = embedding_dict['embeddings.position_embedding.weight'].cuda().type(data_type)
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
        token_type_ids,
    ):
        if token_type_ids is None:
            token_type_ids = self.token_type_ids
        return self.embedding.forward_transformers(input_ids, position_ids, token_type_ids, self.if_layernorm)

    @staticmethod
    def from_torch(config, embedding_dict, data_type=torch.float32):
        embedding = EETCLIPTextEmbedding(config, embedding_dict, data_type=data_type)
        return embedding
    

class EETCLIPTextTransformer():
    def __init__(self, config, embedding, encoder, layernorm):
        self.embedding = embedding
        self.encoder = encoder
        self.final_layer_norm = layernorm
        self.pre_padding_len = torch.empty(0).long()
        self.position_ids = torch.arange(0, config.max_position_embeddings).reshape(1, config.max_position_embeddings).cuda()

    def __call__(
        self,
        input_ids,
        position_ids=None,
        token_type_ids=None,
        attention_mask=None,
    ):
        input_shape = input_ids.size()
        position_ids = self.position_ids[:, :input_shape[1]]

        if token_type_ids is not None:
            token_type_ids = token_type_ids.view(-1, input_shape[-1])
        if attention_mask is None:
            pre_padding_len = self.pre_padding_len
        else:
            # transformers 0 - padding;1 - nopadding
            pre_padding_len = torch.sum(1 - attention_mask, 1).long().cuda()
        embedding_out = self.embedding(input_ids, position_ids, token_type_ids)
        # CLIP's text transformer model uses causal mask, set 'need_sequence_mask' to True.
        # https://github.com/openai/CLIP/blob/cfcffb90e69f37bf2ff1e988237a0fbe41f33c04/clip/model.py#L324        
        encoder_out = self.encoder(
            embedding_out,
            pre_padding_len=pre_padding_len,
            normalize_before=True,
            need_sequence_mask=True
        )
        last_hidden_state = self.final_layer_norm(encoder_out)
        text_output = last_hidden_state[torch.arange(last_hidden_state.shape[0]), input_ids.argmax(dim=-1)]
        
        return text_output
    
    @staticmethod
    def from_torch(config, cfg, model_dict, data_type=torch.float32):
        embedding = EETCLIPTextEmbedding.from_torch(config, model_dict['embeddings'], data_type)
        encoder = EETEncoder.from_torch(config, model_dict, cfg.num_hidden_layers, data_type)
        final_layernorm = EETLayerNorm.from_torch(config, model_dict['layernorm']['text_model.final_layer_norm.weight'], model_dict['layernorm']['text_model.final_layer_norm.bias'], data_type)

        text_model = EETCLIPTextTransformer(cfg, embedding, encoder, final_layernorm)
        return text_model


class EETCLIPModel():
    def __init__(self, config, text_model, vision_model, visual_proj, text_proj, scale):
        self.text_model = text_model
        self.vision_model = vision_model
        self.visual_projection = visual_proj
        self.text_projection = text_proj
        self.logit_scale = nn.Parameter(scale)
        self.config = config

    def __call__(
        self,
        input_ids,
        pixel_values,
        position_ids=None,
        token_type_ids=None,
        attention_mask=None,
    ):
        vision_output = self.vision_model(
            pixel_values=pixel_values,
            position_ids=position_ids
        )

        text_output = self.text_model(
            input_ids=input_ids,
            position_ids=position_ids,
            token_type_ids=token_type_ids,
            attention_mask=attention_mask,
        )

        image_embeds = self.visual_projection(vision_output)
        text_embeds = self.text_projection(text_output)

        # normalized features
        image_embeds = image_embeds / image_embeds.norm(dim=-1, keepdim=True)
        text_embeds = text_embeds / text_embeds.norm(dim=-1, keepdim=True)

        # cosine similarity as logits
        logit_scale = self.logit_scale.exp()
        logits_per_text = torch.matmul(text_embeds, image_embeds.t()) * logit_scale
        logits_per_image = logits_per_text.T

        return (logits_per_image, logits_per_text)

    @staticmethod
    def create_state_dict(torch_model, model_name):
        model_dict = {}
        text_layer_model_dict = {}
        text_embedding_dict = {}
        text_layernorm_dict = {}
        vision_layer_model_dict = {}
        vision_embedding_dict = {}
        vision_layernorm_dict = {}

        for k, v in torch_model.state_dict().items():
            # Structure mapping
            k = convert_name(k, model_name)
            if "text_model" in k:
                if "embeddings." in k:
                    k = k[k.find('embeddings.'):]
                    text_embedding_dict[k] = v
                elif "layer." in k:
                    k = k[k.find('layer.'):]
                    text_layer_model_dict[k] = v
                else:
                    text_layernorm_dict[k] = v
            elif "vision_model" in k:
                if "embeddings." in k:
                    k = k[k.find('embeddings.'):]
                    vision_embedding_dict[k] = v
                elif "layer." in k:
                    k = k[k.find('layer.'):]           
                    vision_layer_model_dict[k] = v
                else:
                    vision_layernorm_dict[k] = v
            else:
                model_dict[k] = v

        # Group by state dict layer id.
        from itertools import groupby
        vision_model_dict = {k: dict(v) for k, v in groupby(list(vision_layer_model_dict.items()),
                                                            lambda item: item[0][:(item[0].index('.', item[0].index('.')+1))])}
        text_model_dict = {k: dict(v) for k, v in groupby(list(text_layer_model_dict.items()),
                                                          lambda item: item[0][:(item[0].index('.', item[0].index('.')+1))])}
        vision_model_dict['embeddings'] = vision_embedding_dict
        vision_model_dict['layernorm'] = vision_layernorm_dict
        text_model_dict['embeddings'] = text_embedding_dict
        text_model_dict['layernorm'] = text_layernorm_dict
        model_dict['vision_model'] = vision_model_dict
        model_dict['text_model'] = text_model_dict

        return model_dict

    @staticmethod
    def from_pretrained(model_id_or_path: str, max_batch, data_type, num_channels=3, device_id=0):
        """from_pretrained."""
        torch.set_grad_enabled(False)
        torch_model = CLIPModel.from_pretrained(model_id_or_path)
        cfg = torch_model.config
        text_cfg = cfg.text_config
        vision_cfg = cfg.vision_config
        model_name = cfg.model_type

        # torch model config 'num_channels' is required but not set 
        vision_cfg.num_channels = num_channels

        # create eet model state dict
        model_dict = EETCLIPModel.create_state_dict(torch_model, model_name)

        device = "cpu" if device_id < 0 else f"cuda:{device_id}"
        batch_size = max_batch
        text_config = meta_desc(batch_size, text_cfg.num_attention_heads, text_cfg.hidden_size,
                                text_cfg.num_hidden_layers, text_cfg.hidden_size, text_cfg.hidden_size, data_type, device, False, text_cfg.hidden_act)
        vision_config = meta_desc(batch_size, vision_cfg.num_attention_heads, vision_cfg.hidden_size,
                                  vision_cfg.num_hidden_layers, vision_cfg.hidden_size, vision_cfg.hidden_size, data_type, device, False, vision_cfg.hidden_act)

        # vision model        
        vision_model = EETCLIPVisionTransformer.from_torch(vision_config, vision_cfg, model_dict['vision_model'], data_type)
        visual_proj = FCLayer(model_dict['visual_projection.weight'], bias=None, data_type=data_type)

        # text model
        text_model = EETCLIPTextTransformer.from_torch(text_config, text_cfg, model_dict['text_model'], data_type)
        text_proj = FCLayer(model_dict['text_projection.weight'], bias=None, data_type=data_type)

        eet_clip_model = EETCLIPModel(cfg, text_model, vision_model, visual_proj, text_proj, model_dict['logit_scale'])
        return eet_clip_model




        