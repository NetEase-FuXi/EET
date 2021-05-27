#
# Created by djz on 2021/01/21.
#
"""EET transformers gpt2 model. """

import math
import time
import torch
import torch.nn as nn
from torch import Tensor
from typing import Any, Dict, List, Optional, Tuple
from transformers import GPT2Model,GPT2PreTrainedModel
from transformers.configuration_gpt2 import GPT2Config
from transformers.configuration_utils import PretrainedConfig

from EET import MetaDesc as meta_desc
from EET import LayerNorm as eet_layernorm
from EET import FeedForwardNetwork as eet_ffn
from EET import Embedding as eet_embedding
from EET import MaskedMultiHeadAttention as eet_attention
from EET import CrossMultiHeadAttention as eet_cross_attention


__all__ = [
    'EETLayerNorm', 'EETGPT2Embedding', 'EETGPT2Feedforward', 'EETGPT2Attention', 'EETGPT2DecoderLayer', 'EETGPT2Decoder', 'EETGPT2Model'
]

class EETLayerNorm():
    def __init__(self,meta_des,layernorm_weights,layernorm_bias,data_type = torch.float32):
        self.layernorm_weights = layernorm_weights.cuda().type(data_type)
        self.layernorm_bias = layernorm_bias.cuda().type(data_type)
        self.layernorm = eet_layernorm(meta_des,self.layernorm_weights,self.layernorm_bias)

    def __call__(self,
                input_ids):
        return self.layernorm.layer_norm(input_ids)
    
    @staticmethod
    def from_torch(meta_des,layernorm_weights,layernorm_bias,data_type = torch.float32):
        layernorm = EETLayerNorm(meta_des,layernorm_weights,layernorm_bias,data_type = data_type)
        return layernorm

class EETGPT2Embedding():
    def __init__(self,meta_des,embedding_dict,data_type = torch.float32):
        self.embedding_weights = embedding_dict['wte.weight'].cuda().type(data_type)
        self.position_weights = embedding_dict['wpe.weight'].cuda().type(data_type)
        # not use token_type
        self.token_type_ids =  torch.empty(0).long()
        self.token_type_weights = torch.empty(0)
        self.if_layernorm = False
        # not use layernorm
        self.Layernorm_weights = torch.empty(0) 
        self.Layernorm_bias = torch.empty(0)
        self.embedding = eet_embedding(meta_des,self.embedding_weights,self.position_weights,self.token_type_weights,self.Layernorm_weights,self.Layernorm_bias)
    def __call__(self,
                input_ids,
                position_ids,
                token_type_ids):
        if_layernorm = False
        if token_type_ids is None:
            token_type_ids = self.token_type_ids
        return self.embedding.forward_transformers(input_ids,position_ids,token_type_ids,if_layernorm)
    
    @staticmethod
    def from_torch(meta_des,embedding_dict,data_type = torch.float32):
        feedforward = EETGPT2Embedding(meta_des,embedding_dict,data_type = data_type)
        return feedforward

class EETGPT2Feedforward():
    def __init__(self,meta_des,model_dict,layer_id,data_type = torch.float32):
        self.intermediate_weights = torch.clone(torch.t([x[1] for x in model_dict.items() if 'mlp.c_fc.weight' in x[0]][0].transpose(0,1)).contiguous()).contiguous().cuda().type(data_type)
        self.intermediate_bias = [x[1] for x in model_dict.items() if 'mlp.c_fc.bias' in x[0]][0].cuda().type(data_type)
        self.output_weights = torch.clone(torch.t([x[1] for x in model_dict.items() if 'mlp.c_proj.weight' in x[0]][0].transpose(0,1)).contiguous()).contiguous().cuda().type(data_type)
        self.output_bias = [x[1] for x in model_dict.items() if 'mlp.c_proj.bias' in x[0]][0].cuda().type(data_type)
        self.layernorm_weights = [x[1] for x in model_dict.items() if 'ln_2.weight' in x[0]][0].cuda().type(data_type)
        self.layernorm_bias = [x[1] for x in model_dict.items() if 'ln_2.bias' in x[0]][0].cuda().type(data_type)
        self.ffn = eet_ffn(meta_des,self.intermediate_weights,self.intermediate_bias,self.output_weights,self.output_bias,self.layernorm_weights,self.layernorm_bias)
    def __call__(self,
                input_id,
                pre_layernorm = True,
                add_redusial = True):
        return self.ffn.forward(input_id,pre_layernorm,add_redusial)
    
    @staticmethod
    def from_torch(meta_des,model_dict,layer_id,data_type = torch.float32):
        feedforward = EETGPT2Feedforward(meta_des,model_dict,layer_id,data_type = data_type)
        return feedforward

class EETGPT2Attention():
    def __init__(self,meta_des, model_dict,layer_id,data_type = torch.float32,cross_attn = False):
        self.cross_attn = cross_attn
        if self.cross_attn:
            self.layernorm_weights = [x[1] for x in model_dict.items() if 'ln_1.weight' in x[0]][0].cuda().type(data_type)
            self.layernorm_bias = [x[1] for x in model_dict.items() if 'ln_1.bias' in x[0]][0].cuda().type(data_type)
            emb_size = self.layernorm_bias.size()[0]
            self.q_weights = torch.clone(torch.t([x[1] for x in model_dict.items() if 'c_attn.weight' in x[0]][0].transpose(0,1)[:emb_size]).contiguous()).cuda().type(data_type)
            self.k_weights = torch.clone(torch.t([x[1] for x in model_dict.items() if 'c_attn.weight' in x[0]][0].transpose(0,1)[emb_size:emb_size*2]).contiguous()).cuda().type(data_type)
            self.v_weights = torch.clone(torch.t([x[1] for x in model_dict.items() if 'c_attn.weight' in x[0]][0].transpose(0,1)[emb_size*2:]).contiguous()).cuda().type(data_type)
            self.q_bias = [x[1] for x in model_dict.items() if 'c_attn.bias' in x[0]][0][:emb_size].cuda().type(data_type)
            self.k_bias = [x[1] for x in model_dict.items() if 'c_attn.bias' in x[0]][0][emb_size:emb_size*2].cuda().type(data_type)
            self.v_bias = [x[1] for x in model_dict.items() if 'c_attn.bias' in x[0]][0][emb_size*2:].cuda().type(data_type)
            self.out_weights = [x[1] for x in model_dict.items() if 'attn.c_proj.weight' in x[0]][0].cuda().type(data_type)
            self.out_bias = [x[1] for x in model_dict.items() if 'attn.c_proj.bias' in x[0]][0].cuda().type(data_type)
            self.attention = eet_cross_attention(meta_des,self.q_weights,self.k_weights,self.v_weights,self.q_bias,self.k_bias,self.v_bias,self.out_weights,self.out_bias,self.layernorm_weights,self.layernorm_bias)
        else:
            self.layernorm_weights = [x[1] for x in model_dict.items() if 'ln_1.weight' in x[0]][0].cuda().type(data_type)
            self.layernorm_bias = [x[1] for x in model_dict.items() if 'ln_1.bias' in x[0]][0].cuda().type(data_type)
            emb_size = self.layernorm_bias.size()[0]
            self.q_weights = torch.clone(torch.t([x[1] for x in model_dict.items() if 'c_attn.weight' in x[0]][0].transpose(0,1)[:emb_size]).contiguous()).cuda().type(data_type)
            self.k_weights = torch.clone(torch.t([x[1] for x in model_dict.items() if 'c_attn.weight' in x[0]][0].transpose(0,1)[emb_size:emb_size*2]).contiguous()).cuda().type(data_type)
            self.v_weights = torch.clone(torch.t([x[1] for x in model_dict.items() if 'c_attn.weight' in x[0]][0].transpose(0,1)[emb_size*2:]).contiguous()).cuda().type(data_type)
            self.q_bias = [x[1] for x in model_dict.items() if 'c_attn.bias' in x[0]][0][:emb_size].cuda().type(data_type)
            self.k_bias = [x[1] for x in model_dict.items() if 'c_attn.bias' in x[0]][0][emb_size:emb_size*2].cuda().type(data_type)
            self.v_bias = [x[1] for x in model_dict.items() if 'c_attn.bias' in x[0]][0][emb_size*2:].cuda().type(data_type)
            self.out_weights = [x[1] for x in model_dict.items() if 'attn.c_proj.weight' in x[0]][0].cuda().type(data_type)
            self.out_bias = [x[1] for x in model_dict.items() if 'attn.c_proj.bias' in x[0]][0].cuda().type(data_type)
            self.attention = eet_attention(meta_des,self.q_weights,self.k_weights,self.v_weights,self.q_bias,self.k_bias,self.v_bias,self.out_weights,self.out_bias,self.layernorm_weights,self.layernorm_bias)

    def __call__(self,
                input_id,
                pre_padding_len,
                reorder_state = None,
                encoder_out = None,
                encoder_padding_mask  = None,
                pre_layernorm = False,
                add_redusial = True,
                first_pass = False):
                
        if self.cross_attn:
            return self.attention.forward(input_id,encoder_out,pre_padding_len,pre_layernorm,add_redusial,encoder_padding_mask,first_pass)
        else:
            return self.attention.forward(input_id,pre_padding_len,reorder_state,pre_layernorm,add_redusial, first_pass)

    @staticmethod
    def from_torch(meta_des,model_dict,layer_id,data_type = torch.float32):
        attention = EETGPT2Attention(meta_des,model_dict,layer_id,data_type = data_type)
        return attention

class EETGPT2DecoderLayer():
    def __init__(self, config, attention,feedforward,cross_attention = None):
        self.attetion = attention
        self.cross_attention = cross_attention
        self.feedforward = feedforward
        self.normalize_before = True
        self.add_redusial = True
        self.add_cross_attention = config.add_cross_attention
    def __call__(self,
                x,
                encoder_out = None,
                first_pass = True,
                pre_padding_len = None,
                encoder_attention_mask = None,
                head_mask = None,
                reorder_state = None):

        ''' gpt2 model struct '''
        ''' layernorm->self_attention-> project->addinputbias->layernorm->ffn->addinputbias'''
        if encoder_out is not None and self.cross_attention is not None:
            self_attn_out = self.attetion(input_id = x,
                        pre_padding_len = pre_padding_len,
                        reorder_state = reorder_state,
                        pre_layernorm = self.normalize_before,
                        add_redusial = self.add_redusial,
                        first_pass = first_pass)

            cross_attn_out = self.cross_attention(input_id = self_attn_out,
                        pre_padding_len = pre_padding_len,
                        encoder_out = encoder_out,
                        encoder_padding_mask = encoder_attention_mask,
                        pre_layernorm = self.normalize_before,
                        add_redusial = self.add_redusial,
                        first_pass = first_pass)

            out = self.feedforward(cross_attn_out,
                        pre_layernorm = self.normalize_before,
                        add_redusial = self.add_redusial)
        else:
            self_attn_out = self.attetion(input_id = x,
                        pre_padding_len = pre_padding_len,
                        reorder_state = reorder_state,
                        pre_layernorm = self.normalize_before,
                        add_redusial = self.add_redusial,
                        first_pass = first_pass)

            out = self.feedforward(self_attn_out,
                        pre_layernorm = self.normalize_before,
                        add_redusial = self.add_redusial)          

        return out

    @staticmethod
    def from_torch(meta_des, config,model_dict,layer_id,data_type = torch.float32):
        attention = EETGPT2Attention.from_torch(meta_des = meta_des, model_dict = model_dict, layer_id = layer_id,data_type = data_type)
        feedforward = EETGPT2Feedforward.from_torch(meta_des = meta_des, model_dict = model_dict, layer_id = layer_id,data_type = data_type)

        if config.add_cross_attention:
            cross_attention = EETGPT2Attention.from_torch(meta_des = meta_des, model_dict = model_dict, layer_id = layer_id,data_type = data_type)
            layer = EETGPT2DecoderLayer(config, attention, feedforward,cross_attention)
        else:
            layer = EETGPT2DecoderLayer(config, attention, feedforward)
        
        return layer

class EETGPT2Decoder():
    def __init__(self,DecoderLayers):
        self.layers = DecoderLayers
    def __call__(
        self,
        x,
        encoder_out = None,
        first_pass = True,
        pre_padding_len = None,
        encoder_attention_mask = None,
        head_mask = None,
        reorder_state = None,
    ):
        for layer in self.layers:
            x = layer(x,
                    encoder_out = encoder_out,
                    first_pass = first_pass,
                    pre_padding_len = pre_padding_len,
                    encoder_attention_mask = encoder_attention_mask,
                    head_mask = None,
                    reorder_state = reorder_state)
        return x
    
    @staticmethod
    def from_torch(layer_model_dict,meta_des,config,data_type = torch.float32):
        """from torch."""
        DecoderLayers = []
        for i in range(config.n_layer):
            if i < 10:
                DecoderLayers.extend(
                    [
                        EETGPT2DecoderLayer.from_torch(meta_des,config,layer_model_dict['h.'+str(i)+'.'],i,data_type = data_type)
                    ]
                )
            else:
                DecoderLayers.extend(
                    [
                        EETGPT2DecoderLayer.from_torch(meta_des,config,layer_model_dict['h.'+str(i)],i,data_type = data_type)
                    ]
                )

        eet_decoder =  EETGPT2Decoder(DecoderLayers)
        return eet_decoder

class EETGPT2Model():
    def __init__(self,config,embedding,decoder, layer_norm):
        self.embedding = embedding
        self.decoder = decoder
        self.layer_norm = layer_norm
        self.position_ids = torch.arange(0,config.n_positions).reshape(1,config.n_positions).cuda()
        self.self_attn_padding_mask = torch.empty(0).long()
        self.encoder_attention_mask = torch.empty(0)
        self.reorder_state = torch.empty(0).long()

    def __call__(
        self,
        input_ids,
        encoder_out = None,
        first_pass = True,
        position_ids = None,
        token_type_ids = None,
        attention_mask = None,
        reorder_state = None,
    ):
        """ EET suport left padding, ``0`` for tokens that are NOT MASKED, ``1`` for MASKED tokens. The struct like [1,1,1,0,0,0]"""

        input_shape = input_ids.size()
        batch_size = input_shape[0]
        # Attention mask.
        if attention_mask is  None:
            pre_padding_len = self.self_attn_padding_mask
        else:
            pre_padding_len = torch.sum(attention_mask, 1, True).cuda().long()

        position_ids = self.position_ids[:, :input_shape[1]]
        if token_type_ids is not None:
            token_type_ids = token_type_ids.view(-1, input_shape[-1])
        embedding_out = self.embedding(input_ids,position_ids,token_type_ids)

        if reorder_state is not None:
            self.reorder_state = reorder_state.long()

        decoder_out = self.decoder(embedding_out,
                    encoder_out = encoder_out,
                    first_pass = first_pass,
                    pre_padding_len = pre_padding_len,
                    encoder_attention_mask = self.encoder_attention_mask,
                    head_mask = None,
                    reorder_state = self.reorder_state,)
        
        decoder_out = self.layer_norm(decoder_out)
        return decoder_out
    
    @staticmethod
    def from_pretrained(model_id_or_path: str,max_batch,full_seq_len,data_type = torch.float32):
        """from torch."""
        torch.set_grad_enabled(False)
        model_dict = {}
        embedding_dict = {}
        layernorm_dict = {}

        torch_model = GPT2Model.from_pretrained(model_id_or_path)
        cfg = torch_model.config

        for k, v in torch_model.state_dict().items():
            if 'e.' in k:
                embedding_dict[k] = v
            if 'h.' in k:
                model_dict[k] = v
            if 'ln_f' in k:
                layernorm_dict[k] = v

        from itertools import groupby

        layer_model_dict = {k: dict(v) for k, v in groupby(list(model_dict.items()), lambda item: item[0][:4])}

        # data_type = torch.float32
        device = "cuda:0"
        activation_fn = cfg.activation_function
        batch_size = max_batch
        full_seq_len = full_seq_len
        meta_des = meta_desc(batch_size, cfg.n_head, cfg.n_embd, cfg.n_layer , cfg.n_positions,full_seq_len, data_type, device, False, activation_fn)
        layer_norm = EETLayerNorm.from_torch(meta_des,layernorm_dict['ln_f.weight'],layernorm_dict['ln_f.bias'],data_type)
        embedding = EETGPT2Embedding.from_torch(meta_des,embedding_dict,data_type)
        # embedding = None
        decoder = EETGPT2Decoder.from_torch(layer_model_dict,meta_des, cfg,data_type)
        eet_model =  EETGPT2Model(cfg,embedding, decoder,layer_norm)
        return eet_model
