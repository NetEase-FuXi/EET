#
# Created by zsd on 2023/06/20.
#
"""EET llama model. """

import os
import math
import time
import copy
import numpy as np
from tqdm import tqdm
import torch
import torch.nn as nn
import torch.nn.functional as F
from torch import Tensor
from torch.nn.parameter import Parameter
from typing import Any, Dict, List, Optional, Tuple
from transformers import LlamaModel
from transformers.modeling_outputs import (
    BaseModelOutput,
    BaseModelOutputWithPastAndCrossAttentions,
    Seq2SeqLMOutput,
    Seq2SeqModelOutput,
    CausalLMOutputWithPast,
)
from transformers.generation import GenerationConfig
from transformers.generation.logits_process import LogitsProcessor
from transformers.generation.utils import LogitsProcessorList, StoppingCriteriaList, GenerationConfig, ModelOutput
from transformers import AutoModel, AutoModelForCausalLM

from eet.transformers.encoder_decoder import *
from eet.utils.mapping import convert_name
from torch.nn import BCEWithLogitsLoss, CrossEntropyLoss, MSELoss

from EET import MetaDesc as meta_desc
from EET import Embedding as eet_embedding
from EET import GatedFeedForwardNetworkInt8 as eet_gated_ffn
from EET import GatedFeedForwardNetwork as eet_gated_ffn_fp16
from EET import LlamaMmha as eet_masked_attention

from EET import preprocess_weights as preprocess_weights
from EET import quant_weights as quant_and_preprocess_weights

from ..pipelines.generation import GenerationMixin_EET
import logging
logger = logging.getLogger(__name__)


__all__ = ['EETLlamaSelfMaskedAttention','EETGatedFeedForward', 'EETLlamaDecoderLayer', 'EETLlamaDecoder', 'EETLlamaModel','EETLlamaForCausalLM', 'EETRewardModel', 'convert_llama_weights']

class EETLlamaSelfMaskedAttention():
    def __init__(self, config, model_dict, layer_id, data_type=torch.float32, bias=True, is_int8=True):
        self.q_bias = torch.empty(0)
        self.k_bias = torch.empty(0)
        self.v_bias = torch.empty(0)
        self.layernorm_bias = torch.empty(0)
        self.out_bias = torch.empty(0)

        self.layernorm_weights = model_dict['layer.' + str(layer_id) + '.self_attn.layernorm.weight'].cuda().type(data_type)
        self.qkv_weights = model_dict['layer.' + str(layer_id) + '.self_attn.qkv_proj.weight'].cuda()
        self.out_weights = model_dict['layer.' + str(layer_id) + '.self_attn.out_proj.weight'].cuda()

        self.qkv_scale = model_dict['layer.' + str(layer_id) + '.self_attn.qkv_proj.scale'].half().cuda() if is_int8 else torch.empty(0)
        self.out_scale = model_dict['layer.' + str(layer_id) + '.self_attn.out_proj.scale'].half().cuda() if is_int8 else torch.empty(0)
        self.attention = eet_masked_attention(config, self.qkv_weights,self.qkv_scale, self.q_bias, self.k_bias,
                                              self.v_bias, self.out_weights,self.out_scale, self.out_bias, self.layernorm_weights, self.layernorm_bias)

    def __call__(
        self,
        hidden_states,
        pre_padding_len,
        reorder_state=None,
        pre_layernorm=False,
        add_residual=True,
        first_pass=False,
        relative_attention_bias=torch.empty(0),
    ):
        return self.attention.forward(hidden_states, pre_padding_len, reorder_state, pre_layernorm, add_residual, first_pass, relative_attention_bias)

    def update(self, model_dict, layer_id, data_type):
        self.layernorm_weights.copy_(model_dict['layer.' + str(layer_id) + '.self_attn.layernorm.weight'].cuda().type(data_type).detach())
        q_weights = model_dict['layer.' + str(layer_id) + '.self_attn.q_proj.weight']
        k_weights = model_dict['layer.' + str(layer_id) + '.self_attn.k_proj.weight']
        v_weights = model_dict['layer.' + str(layer_id) + '.self_attn.v_proj.weight']
        unprocessed_qkv_weights = torch.cat((q_weights, k_weights, v_weights), 0).transpose(0, 1).contiguous().cpu()
        qkv_weights, qkv_scale = quant_and_preprocess_weights(unprocessed_qkv_weights, torch.int8, False)
        self.qkv_weights.copy_(qkv_weights.cuda().detach())
        self.qkv_scale.copy_(qkv_scale.half().cuda().detach())

        unprocessed_out_weights = torch.t(model_dict['layer.' + str(layer_id) + '.self_attn.out_proj.weight']).contiguous().cpu()
        out_weights, out_scale = quant_and_preprocess_weights(unprocessed_out_weights, torch.int8, False)
        self.out_weights.copy_(out_weights.cuda().detach())
        self.out_scale.copy_(out_scale.half().cuda().detach())

    @staticmethod
    def from_eet(config, model_dict, layer_id, data_type=torch.float32, bias=True, is_int8=True):
        attention = EETLlamaSelfMaskedAttention(config, model_dict, layer_id, data_type=data_type, bias=bias, is_int8=is_int8)
        return attention

 
class EETGatedFeedForward():
    def __init__(self, config, model_dict, layer_id, data_type=torch.float32, bias=False, name="out_cache", is_int8=False):
        self.is_int8 = is_int8
        self.intermediate_0_bias = torch.empty(0)
        self.intermediate_1_bias = torch.empty(0)
        self.output_bias = torch.empty(0)
        self.layernorm_bias = torch.empty(0)

        self.intermediate_0_weights = model_dict['layer.' + str(layer_id) + '.ffn.intermediate_0.weight'].cuda()
        self.intermediate_0_scale = model_dict['layer.' + str(layer_id) + '.ffn.intermediate_0.scale'].half().cuda() if is_int8 else torch.empty(0)
        self.intermediate_1_weights = model_dict['layer.' + str(layer_id) + '.ffn.intermediate_1.weight'].cuda()
        self.intermediate_1_scale = model_dict['layer.' + str(layer_id) + '.ffn.intermediate_1.scale'].half().cuda() if is_int8 else torch.empty(0)
        self.output_weights = model_dict['layer.' + str(layer_id) + '.ffn.output.weight'].cuda()
        self.output_scale = model_dict['layer.' + str(layer_id) + '.ffn.output.scale'].half().cuda() if is_int8 else torch.empty(0)

        self.layernorm_weights = model_dict['layer.' + str(layer_id) + '.ffn.layernorm.weight'].cuda().type(data_type)

        if self.is_int8:
            self.ffn = eet_gated_ffn(config, self.intermediate_0_weights, self.intermediate_0_scale, self.intermediate_0_bias, self.intermediate_1_weights,
                                    self.intermediate_1_scale, self.intermediate_1_bias, self.output_weights, self.output_scale, self.output_bias, self.layernorm_weights, self.layernorm_bias, name)
        else:
            self.ffn = eet_gated_ffn_fp16(config, self.intermediate_0_weights, self.intermediate_0_bias, self.intermediate_1_weights, self.intermediate_1_bias, self.output_weights, self.output_bias,
                                          self.layernorm_weights, self.layernorm_bias, name)

    def __call__(
        self,
        hidden_states,
        pre_layernorm=True,
        add_residual=True
    ):
        return self.ffn.forward(hidden_states, pre_layernorm, add_residual)

    def update(self, model_dict, layer_id, data_type):
        self.layernorm_weights.copy_(model_dict['layer.' + str(layer_id) + '.ffn.layernorm.weight'].cuda().type(data_type).detach())
        unprocessed_intermediate_0_weights = torch.t(model_dict['layer.' + str(layer_id) + '.ffn.intermediate_0.weight']).contiguous().cpu()
        intermediate_0_weights, intermediate_0_scale = quant_and_preprocess_weights(unprocessed_intermediate_0_weights, torch.int8, False)
        self.intermediate_0_weights.copy_(intermediate_0_weights.cuda().detach())
        self.intermediate_0_scale.copy_(intermediate_0_scale.half().cuda().detach())

        unprocessed_intermediate_1_weights = torch.t(model_dict['layer.' + str(layer_id) + '.ffn.intermediate_1.weight']).contiguous().cpu()
        intermediate_1_weights, intermediate_1_scale = quant_and_preprocess_weights(unprocessed_intermediate_1_weights, torch.int8, False)
        self.intermediate_1_weights.copy_(intermediate_1_weights.cuda().detach())
        self.intermediate_1_scale.copy_(intermediate_1_scale.half().cuda().detach())

        unprocessed_output_weights = torch.t(model_dict['layer.' + str(layer_id) + '.ffn.output.weight']).contiguous().cpu()
        output_weights, output_scale = quant_and_preprocess_weights(unprocessed_output_weights, torch.int8, False)
        self.output_weights.copy_(output_weights.cuda().detach())
        self.output_scale.copy_(output_scale.half().cuda().detach())

    @staticmethod
    def from_eet(config, model_dict, layer_id, data_type=torch.float32, bias=True, name="out_cache", is_int8=False):
        feedforward = EETGatedFeedForward(config, model_dict, layer_id, data_type=data_type, bias=bias, name=name, is_int8=is_int8)
        return feedforward


class EETLlamaDecoderLayer():
    def __init__(self, config, attention, feedforward):
        self.attention = attention
        self.feedforward = feedforward

    def __call__(
        self,
        hidden_states,
        position_ids=None,
        first_pass=True,
        pre_padding_len=None,
        head_mask=None,
        reorder_state=None,
        normalize_before=True,
        add_residual=True,
    ):

        ''' self_attn -> ffn'''
        self_attn_out = self.attention(
            hidden_states=hidden_states,
            pre_padding_len=pre_padding_len,
            reorder_state=reorder_state,
            pre_layernorm=normalize_before,
            add_residual=add_residual,
            first_pass=first_pass
        )

        out = self.feedforward(
            self_attn_out,
            pre_layernorm=normalize_before,
            add_residual=add_residual
        )
        return out
    
    def update(self, model_dict, layer_id, data_type=torch.float32):
        self.attention.update(model_dict, layer_id, data_type)
        self.feedforward.update(model_dict, layer_id, data_type)

    @staticmethod
    def from_eet(config, model_dict, layer_id, data_type=torch.float32, bias=True, is_int8=True):
        attention = EETLlamaSelfMaskedAttention.from_eet(config, model_dict, layer_id, data_type=data_type, bias=bias, is_int8=is_int8)
        feedforward = EETGatedFeedForward.from_eet(config, model_dict, layer_id, data_type=data_type, bias=bias, name="decoder_out_cache", is_int8=is_int8)

        layer = EETLlamaDecoderLayer(config, attention, feedforward)
        return layer


class EETLlamaDecoder():
    def __init__(self, DecoderLayers):
        self.layers = DecoderLayers

    def __call__(
        self,
        hidden_states,
        position_ids=None,
        first_pass=True,
        pre_padding_len=None,
        reorder_state=None,
        normalize_before=True,
        context_lengths=None,
    ):
        for layer in self.layers:
                hidden_states = layer(
                hidden_states,
                position_ids=None,
                first_pass=first_pass,
                pre_padding_len=pre_padding_len,
                reorder_state=reorder_state,
                normalize_before=normalize_before,
                add_residual=True
            )

        return hidden_states
    
    def update(self, layer_model_dict, layer_num, data_type=torch.float32):
        for i in tqdm(range(layer_num), desc="[EET][INFO] model weight updating..."):
            self.layers[i].update(layer_model_dict[str(i)], i, data_type=data_type)

    @staticmethod
    def from_eet(config, layer_model_dict, layer_num, data_type=torch.float32, bias=True, is_int8=True):
        """from torch."""
        DecoderLayers = []
        for i in tqdm(range(layer_num), desc="[EET][INFO] loading weight..."):
            DecoderLayers.extend(
                [
                    EETLlamaDecoderLayer.from_eet(config, layer_model_dict[str(i)], i, data_type=data_type, bias=bias, is_int8=is_int8)
                ]
            )

        eet_decoder = EETLlamaDecoder(DecoderLayers)
        return eet_decoder


class EETLlamaModel():
    def __init__(self, config, decoder, layer_norm, embedding=None, max_batch=1, max_prompt_len=1024):
        self.embedding = embedding
        self.decoder = decoder
        self.layer_norm = layer_norm
        self.pre_padding_len = torch.empty(0).long()
        self.reorder_state = torch.empty(0).long()
        self.context_lengths = torch.empty(0).long()
        self.current_len = 0
        self.config = config
        self.max_batch = max_batch
        self.max_prompt_len = max_prompt_len

    def __call__(
        self,
        input_ids,
        first_pass=True,
        position_ids=None,
        token_type_ids=None,
        attention_mask=None,
        reorder_state=None,
        context_lengths=None,
    ):
        input_shape = input_ids.size()
        assert input_shape[0] <= self.max_batch, "[EET][ERROR] batch size must less than max_batch {}, but get batch {}".format(self.max_batch, input_shape[0])
        assert input_shape[1] <= self.max_prompt_len, "[EET][ERROR] input_ids length must less than max_prompt_len {}, but get prompt_len {}".format(self.max_prompt_len, input_shape[1])
        # Attention mask.
        if attention_mask is None:
            pre_padding_len = self.pre_padding_len
        else:
            pre_padding_len = torch.sum(1 - attention_mask, 1, True).cuda().long()

        if reorder_state is not None:
            self.reorder_state = reorder_state.long()

        embedding_out = self.embedding(input_ids)   # [batch, seq, hidden_size]

        decoder_out = self.decoder(
            embedding_out,
            position_ids=position_ids,
            first_pass=first_pass,
            pre_padding_len=pre_padding_len,
            reorder_state=self.reorder_state,
            normalize_before=True,
            context_lengths=context_lengths,
        )

        decoder_out = self.layer_norm(decoder_out)
        return decoder_out

    def update(self, torch_model, data_type=torch.float16, model_attr="model"):
        torch.set_grad_enabled(False)
        model_dict = {}
        embedding_dict = {}
        layernorm_dict = {}
        decoder_dict = {}

        cfg = torch_model.config
        base_model = getattr(torch_model, model_attr) if model_attr is not None else torch_model

        for k, v in base_model.state_dict().items():
            if 'embed_tokens.' in k:
                k = k[k.find('weight'):]
                embedding_dict[k] = v
            if 'layers.' in k:
                k = convert_name(k, "llama")
                k = k[k.find('layer.'):]
                model_dict[k] = v
            if 'norm.' in k:
                k = k[k.find('weight'):]
                layernorm_dict[k] = v

        from itertools import groupby
        decoder_list = sorted(decoder_dict.items(), key=lambda item: int(item[0].split('.')[1]))
        layer_model_dict = {k: dict(v) for k, v in groupby(decoder_list, lambda item: item[0].split('.')[1])}
        
        self.embedding.load_state_dict(embedding_dict)
        self.decoder.update(layer_model_dict, cfg.num_hidden_layers, data_type)
        self.layer_norm.update(layernorm_dict, data_type)

    @staticmethod
    def from_eet(llama_dict, cfg, max_batch, max_prompt_seq_len, max_full_seq_len, data_type=torch.float16, device_id=0):
        """from torch."""
        torch.set_grad_enabled(False)
        embedding_dict = {}
        layernorm_dict = {}
        decoder_dict = {}
        is_int8 = True if data_type == torch.int8 else False
        data_type = torch.float16 if data_type == torch.int8 else data_type

        for k, v in llama_dict.items():
            if 'embed_tokens.' in k:
                k = k[k.find('weight'):]
                embedding_dict[k] = v
            if 'layer.' in k:
                decoder_dict[k] = v
            if 'norm.' in k:
                k = k[k.find('norm.'):]
                layernorm_dict[k] = v

        from itertools import groupby
        decoder_list = sorted(decoder_dict.items(), key=lambda item: int(item[0].split('.')[1]))
        layer_model_dict = {k: dict(v) for k, v in groupby(decoder_list, lambda item: item[0].split('.')[1])}

        device = "cpu" if device_id < 0 else f"cuda:{device_id}"
        batch_size = max_batch
        activation_fn = cfg.hidden_act

        meta_des = meta_desc(dtype=data_type,
                             batch_size=batch_size,
                             head_num=cfg.num_attention_heads,
                             hidden_units=cfg.hidden_size,
                             layer_num=cfg.num_hidden_layers,
                             max_seq_len=max_prompt_seq_len,
                             max_full_seq_len=max_full_seq_len,
                             activation_fn=activation_fn,
                             d_ff=cfg.intermediate_size,
                             cuda_device=device,
                             layernorm_eps=cfg.rms_norm_eps,
                             is_int8=is_int8)
        
        embedding = nn.Embedding(cfg.vocab_size, cfg.hidden_size, cfg.pad_token_id)
        embedding.load_state_dict(embedding_dict)
        embedding = embedding.half().cuda()
        decoder = EETLlamaDecoder.from_eet(meta_des, layer_model_dict, layer_num=cfg.num_hidden_layers, data_type=data_type, is_int8=is_int8)
        layer_norm = EETLayerNorm.from_torch(meta_des, layernorm_dict['norm.weight'], None, data_type)

        eet_model = EETLlamaModel(cfg, decoder, layer_norm=layer_norm, embedding=embedding, max_batch=max_batch, max_prompt_len=max_prompt_seq_len)
        return eet_model


    @staticmethod
    def from_pretrained(path_or_file, cfg, max_batch, max_prompt_seq_len, max_full_seq_len, data_type=torch.float32, device_id=0, is_int8=False):
        """from pretrained."""
        torch.set_grad_enabled(False)
        llama_dict = {}
        embedding_dict = {}
        layernorm_dict = {}
        decoder_dict = {}
        if os.path.isfile(path_or_file):
            with open(path_or_file, "rb") as f:
                llama_dict = torch.load(f, map_location=torch.device("cpu"))
        elif os.path.isdir(path_or_file):
            ts_model = AutoModel.from_pretrained(path_or_file, trust_remote_code=True).half()
            llama_dict = convert_llama_weights(ts_model.state_dict(), data_type=data_type)
        else:
            raise ValueError("[EET][ERROR] path_or_file must be a valid path or path to model file, but get {}".format(path_or_file))

        is_int8 = True if data_type == torch.int8 else False
        data_type = torch.float16 if data_type == torch.int8 else data_type

        for k, v in llama_dict.items():
            if 'embed_tokens.' in k:
                k = k[k.find('weight'):]
                embedding_dict[k] = v
            if 'layer.' in k:
                decoder_dict[k] = v
            if 'norm.' in k:
                k = k[k.find('norm.'):]
                layernorm_dict[k] = v

        from itertools import groupby
        decoder_list = sorted(decoder_dict.items(), key=lambda item: int(item[0].split('.')[1]))
        layer_model_dict = {k: dict(v) for k, v in groupby(decoder_list, lambda item: item[0].split('.')[1])}

        device = "cpu" if device_id < 0 else f"cuda:{device_id}"
        batch_size = max_batch
        activation_fn = cfg.hidden_act

        meta_des = meta_desc(dtype=data_type,
                             batch_size=batch_size,
                             head_num=cfg.num_attention_heads,
                             hidden_units=cfg.hidden_size,
                             layer_num=cfg.num_hidden_layers,
                             max_seq_len=max_prompt_seq_len,
                             max_full_seq_len=max_full_seq_len,
                             activation_fn=activation_fn,
                             d_ff=cfg.intermediate_size,
                             cuda_device=device,
                             layernorm_eps=cfg.rms_norm_eps,
                             is_int8=is_int8)
        
        embedding = nn.Embedding(cfg.vocab_size, cfg.hidden_size, cfg.pad_token_id)
        embedding.load_state_dict(embedding_dict)
        embedding = embedding.half().cuda()
        decoder = EETLlamaDecoder.from_eet(meta_des, layer_model_dict, layer_num=cfg.num_hidden_layers, data_type=data_type, is_int8=is_int8)
        layer_norm = EETLayerNorm.from_torch(meta_des, layernorm_dict['norm.weight'], None, data_type)

        eet_model = EETLlamaModel(cfg, decoder, layer_norm=layer_norm, embedding=embedding, max_batch=max_batch, max_prompt_len=max_prompt_seq_len)
        return eet_model


class EETLlamaForCausalLM(GenerationMixin_EET):
    def __init__(self, config, llamamodel, lm_head=None):
        self.config = config
        self.model = llamamodel
        self.main_input_name = "input_ids"
        self.generation_config = GenerationConfig.from_model_config(config)
        self.lm_head = lm_head

    def forward(self):
        NotImplementedError

    def __call__(
        self,
        input_ids,
        first_pass=True,
        position_ids=None,
        attention_mask=None,
        reorder_state=None,
        labels=None,
        **kwargs,
    ):
        outputs = self.model(
            input_ids=input_ids,
            first_pass=first_pass,
            position_ids=position_ids,
            attention_mask=attention_mask,
            reorder_state=reorder_state,
        )

        hidden_states = outputs
        logits = self.lm_head(hidden_states)
        loss = None
        if labels is not None:
            # Shift so that tokens < n predict n
            shift_logits = logits[..., :-1, :].contiguous()
            shift_labels = labels[..., 1:].contiguous()
            # Flatten the tokens
            loss_fct = CrossEntropyLoss()
            shift_logits = shift_logits.view(-1, self.config.vocab_size)
            shift_labels = shift_labels.view(-1)
            # Enable model parallelism
            shift_labels = shift_labels.to(shift_logits.device)
            loss = loss_fct(shift_logits, shift_labels)

        return CausalLMOutputWithPast(
            loss=loss,
            logits=logits,
            past_key_values=None,
            hidden_states=outputs,
            attentions=None,
        )

    def prepare_inputs_for_generation(
        self, input_ids, first_pass=True, past_key_values=None, attention_mask=None, **kwargs
    ):
        if not first_pass:
            input_ids = input_ids[:, -1:]

        position_ids = kwargs.get("position_ids", None)
        if attention_mask is not None and position_ids is None:
            # create position_ids on the fly for batch generation
            position_ids = attention_mask.long().cumsum(-1) - 1
            position_ids.masked_fill_(attention_mask == 0, 1)
            # pre_padding_len = torch.sum(1 - attention_mask, 1).int().cuda()
            if not first_pass:
                position_ids = position_ids[:, -1].unsqueeze(-1)

        model_inputs = {"input_ids": input_ids}

        model_inputs.update(
            {
                "position_ids": position_ids,
                "attention_mask": attention_mask,
            }
        )
        return model_inputs

    def update(self, torch_model, data_type=torch.float16, model_attr="model"):
        """update"""
        torch.set_grad_enabled(False)
        self.lm_head.load_state_dict(torch_model.lm_head.state_dict())
        self.model.update(torch_model, data_type=data_type, model_attr=model_attr)

    @staticmethod
    def from_eet(model_dict, config, max_batch, max_prompt_seq_len, max_full_seq_len, data_type=torch.float32):
        """from torch."""
        torch.set_grad_enabled(False)

        llama_dict = {}
        lm_head_dict = {}
        for k, v in model_dict.items():
            if 'lm_head' in k:
                k = k[k.find('weight'):]
                lm_head_dict[k] = v
            else:
                llama_dict[k] = v

        llamamodel = EETLlamaModel.from_eet(llama_dict, config, max_batch=max_batch, max_prompt_seq_len=max_prompt_seq_len,
                                              max_full_seq_len=max_full_seq_len, data_type=data_type)
        lm_head = nn.Linear(config.hidden_size, config.vocab_size, bias=False)
        lm_head.load_state_dict(lm_head_dict)
        lm_head = lm_head.half().cuda()
        eet_model = EETLlamaForCausalLM(config, llamamodel, lm_head)

        return eet_model

    @staticmethod
    def from_pretrained(path_or_file, config, max_batch, max_prompt_seq_len, max_full_seq_len, data_type=torch.float32):
        """from pretrained."""
        torch.set_grad_enabled(False)
        # cfg = torch_model.config

        model_dict = {}
        llama_dict = {}
        lm_head_dict = {}
        if os.path.isfile(path_or_file):
            with open(path_or_file, "rb") as f:
                model_dict = torch.load(f, map_location=torch.device("cpu"))
        elif os.path.isdir(path_or_file):
            ts_model = AutoModelForCausalLM.from_pretrained(path_or_file, trust_remote_code=True).half()
            model_dict = convert_llama_weights(ts_model.state_dict(), data_type=data_type)
        else:
            raise ValueError("[EET][ERROR] path_or_file must be a valid path or path to model file, but get {}".format(path_or_file))

        for k, v in model_dict.items():
            if 'lm_head' in k:
                k = k[k.find('weight'):]
                lm_head_dict[k] = v
            else:
                llama_dict[k] = v

        llamamodel = EETLlamaModel.from_eet(llama_dict, config, max_batch=max_batch, max_prompt_seq_len=max_prompt_seq_len,
                                              max_full_seq_len=max_full_seq_len, data_type=data_type)
        lm_head = nn.Linear(config.hidden_size, config.vocab_size, bias=False)
        lm_head.load_state_dict(lm_head_dict)
        lm_head = lm_head.half().cuda()
        eet_model = EETLlamaForCausalLM(config, llamamodel, lm_head)

        return eet_model


class EETRewardModel():
    def __init__(self, config, tokenizer, llama, v_head=None, num_padding_at_beginning=0):
        self.config = config
        self.num_padding_at_beginning = num_padding_at_beginning
        self.PAD_ID = tokenizer.pad_token_id

        self.rwtranrsformer = llama
        self.v_head = v_head

    def forward(
        self,
        input_ids,
        first_pass=True,
        position_ids=None,
        attention_mask=None,
        reorder_state=None,
        **kwargs,
    ):
        loss = None

        transformer_outputs = self.rwtranrsformer(
            input_ids=input_ids,
            first_pass=first_pass,
            position_ids=position_ids,
            attention_mask=attention_mask,
            reorder_state=reorder_state,
        )

        hidden_states = transformer_outputs
        rewards = self.v_head(hidden_states).squeeze(-1)
        chosen_mean_scores = []
        rejected_mean_scores = []

        # Split the inputs and rewards into two parts, chosen and rejected
        assert len(input_ids.shape) == 2
        bs = input_ids.shape[0] // 2
        seq_len = input_ids.shape[1]

        chosen_ids = input_ids[:bs]  # bs x seq x 1
        rejected_ids = input_ids[bs:]
        chosen_rewards = rewards[:bs]
        rejected_rewards = rewards[bs:]

        # Compute pairwise loss. Only backprop on the different tokens before padding
        loss = 0
        for i in range(bs):
            chosen_id = chosen_ids[i]
            rejected_id = rejected_ids[i]
            chosen_reward = chosen_rewards[i]
            rejected_reward = rejected_rewards[i]

            c_inds = (chosen_id == self.PAD_ID).nonzero()
            c_ind = c_inds[self.num_padding_at_beginning].item() if len(
                c_inds
            ) > self.num_padding_at_beginning else seq_len  # OPT model pads the first token, so we need to use the second padding token as the end of the sequence
            check_divergence = (chosen_id != rejected_id).nonzero()

            if len(check_divergence) == 0:
                end_ind = rejected_reward.size(-1)
                divergence_ind = end_ind - 1
                r_ind = c_ind
            else:
                # Check if there is any padding otherwise take length of sequence
                r_inds = (rejected_id == self.PAD_ID).nonzero()
                r_ind = r_inds[self.num_padding_at_beginning].item(
                ) if len(r_inds) > self.num_padding_at_beginning else seq_len
                end_ind = max(c_ind, r_ind)
                divergence_ind = check_divergence[0]
            if divergence_ind <= 0:
                print(divergence_ind)
            assert divergence_ind > 0
            c_truncated_reward = chosen_reward[divergence_ind:end_ind]
            r_truncated_reward = rejected_reward[divergence_ind:end_ind]
            chosen_mean_scores.append(
                chosen_reward[c_ind - 1])  #use the end score for reference
            rejected_mean_scores.append(rejected_reward[r_ind - 1])

            # instance_loss = -torch.log(torch.sigmoid(c_truncated_reward - r_truncated_reward)).mean()
            # Fixing the numerical instability
            instance_loss = -torch.nn.functional.logsigmoid(c_truncated_reward - r_truncated_reward).mean()
            # instance_loss = -torch.log(torch.sigmoid(chosen_reward[c_ind - 1] - rejected_reward[r_ind - 1]))
            # instance_loss = -torch.nn.functional.logsigmoid(chosen_reward[c_ind - 1] - rejected_reward[r_ind - 1])
            loss += instance_loss
        loss = loss / bs
        chosen_mean_scores = torch.stack(chosen_mean_scores)
        rejected_mean_scores = torch.stack(rejected_mean_scores)
        return {
            "loss": loss,
            "chosen_mean_scores": chosen_mean_scores,
            "rejected_mean_scores": rejected_mean_scores,
        }

    def forward_value(self,
                      input_ids=None,
                      first_pass=True,
                      attention_mask=None,
                      position_ids=None,
                      return_value_only=False,
                      prompt_length=0,
                      reorder_state=None,
                      use_cache=False):

        transformer_outputs = self.rwtranrsformer(
            input_ids=input_ids,
            first_pass=first_pass,
            position_ids=position_ids,
            attention_mask=attention_mask,
            reorder_state=reorder_state,
        )
        hidden_states = transformer_outputs
        values = self.v_head(hidden_states).squeeze(-1)
        if return_value_only:
            return values
        else:
            # [0 0 0 0 prompt, answer, 0 0 0 0 ] for step 3, we have padding at the beginning
            # [prompt, answer, 0, 0, 0, 0] this is normal
            assert prompt_length > 1, "prompt_length must be greater than 1 to help select the end score"
            bs = values.size(0)
            seq_len = input_ids.shape[1]
            chosen_end_scores = [
            ]  # we use this name for consistency with the original forward function
            for i in range(bs):
                input_id = input_ids[i]
                value = values[i]

                c_inds = (input_id[prompt_length:] == self.PAD_ID).nonzero()
                # here we only use the answer part of the sequence so we do not need to care about the padding at the beginning
                c_ind = c_inds[0].item() + prompt_length if len(
                    c_inds) > 0 else seq_len
                chosen_end_scores.append(value[c_ind - 1])
            return {
                "values": values,
                "chosen_end_scores": torch.stack(chosen_end_scores),
            }

    @staticmethod
    def from_torch(tokenizer, torch_model, max_batch, max_prompt_seq_len, max_full_seq_len, num_padding_at_beginning=0, data_type=torch.float32, device_id=0, model_attr="model", is_int8=False):
        """from torch."""
        torch.set_grad_enabled(False)
        config = torch_model.config
        model_dict = convert_llama_weights(torch_model.state_dict(), data_type=data_type)
        llamamodel = EETLlamaModel.from_eet(model_dict, max_batch=max_batch, max_prompt_seq_len=max_prompt_seq_len,
                                              max_full_seq_len=max_full_seq_len, data_type=data_type, model_attr=model_attr, is_int8=is_int8)

        # torch_model.to(data_type)
        v_head = torch_model.v_head.to(data_type).cuda()
        eet_model = EETRewardModel(config, tokenizer, llamamodel, v_head, num_padding_at_beginning)

        return eet_model

    def update(self, torch_model, data_type=torch.float16, model_attr="model"):
        """update"""
        torch.set_grad_enabled(False)
        self.v_head.load_state_dict(torch_model.v_head.state_dict())
        self.rwtranrsformer.update(torch_model, data_type=data_type, model_attr=model_attr)


def convert_llama_weights(model_dict, data_type=torch.int8, model_attr="model"):
    prefix_len = len(model_attr)
    llama_dict = {}
    qkv_dict = {}
    keys_to_remove = [key for key in model_dict.keys() if 'q_proj' in key or 'k_proj' in key or 'v_proj' in key]

    for k, v in model_dict.items():
        if 'layers.' in k:
            k = k[k.find('layers.'):]
            layer_id = k.split('.')[1]
            if 'q_proj' in k or 'k_proj' in k or 'v_proj' in k:
                qkv_dict.setdefault(layer_id, []).append(v)
    
    for layer_id, qkv_values in qkv_dict.items():
        model_dict[f'layers.{layer_id}.self_attn.qkv_proj.weight'] = torch.cat(qkv_values, dim=0)

    for key in keys_to_remove:
        del model_dict[key]

    if data_type == torch.int8:
        for k, v in tqdm(model_dict.items(), desc="[EET][INFO] int8 model weight quantification and preprocessing..."):
            k = k[prefix_len+1:] if model_attr in k else k
            if 'layers.' in k:
                k = convert_name(k, "llama")
                k = k[k.find('layer.'):]
                if '.weight' in k and v.dim() == 2:
                    unprocessed_weights = v.transpose(0, 1).contiguous()
                    processed_weights, processed_scale = quant_and_preprocess_weights(unprocessed_weights, torch.int8, False)
                    llama_dict[k] = processed_weights                
                    llama_dict[k.replace('.weight', '.scale')] = processed_scale
                else:
                    llama_dict[k] = v
            else:
                llama_dict[k] = v
    else:
        for k, v in tqdm(model_dict.items(), desc="[EET][INFO] fp model weight preprocessing..."):
            k = k[prefix_len+1:] if model_attr in k else k
            if 'layers.' in k:
                k = convert_name(k, "llama")
                k = k[k.find('layer.'):]
                if '.weight' in k and v.dim() == 2:
                    v = v.transpose(0, 1).contiguous()

            llama_dict[k] = v.to(data_type)

    return llama_dict

