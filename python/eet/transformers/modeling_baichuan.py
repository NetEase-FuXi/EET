#
# Created by zsd on 2023/06/20.
#
"""EET baichuan model. """

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
from transformers import AutoModelForCausalLM

from eet.transformers.encoder_decoder import *
from eet.utils.mapping import convert_name
from torch.nn import BCEWithLogitsLoss, CrossEntropyLoss, MSELoss

from EET import MetaDesc as meta_desc
from EET import Embedding as eet_embedding
from EET import GatedFeedForwardNetworkInt8 as eet_gated_ffn
from EET import GatedFeedForwardNetwork as eet_gated_ffn_fp16
from EET import BaichuanMmha as eet_masked_attention
# from FPA_INTB import preprocess_weights as preprocess_weights
from EET import preprocess_weights as preprocess_weights
from EET import quant_weights as quant_and_preprocess_weights

from ..pipelines.generation import GenerationMixin_EET
import logging
logger = logging.getLogger(__name__)


__all__ = ['EETBaichuanSelfMaskedAttention', 'EETBaichuanDecoderLayer', 'EETBaichuanDecoder', 'EETBaichuanModel', 'EETBaichuanForCausalLM', 'convert_baichuan_weights']

def _get_interleave(n):
    def _get_interleave_power_of_2(n):
        start = 2 ** (-(2 ** -(math.log2(n) - 3)))
        ratio = start
        return [start * ratio**i for i in range(n)]

    if math.log2(n).is_integer():
        return _get_interleave_power_of_2(n)
    else:
        closest_power_of_2 = 2 ** math.floor(math.log2(n))
        return (
            _get_interleave_power_of_2(closest_power_of_2)
            + _get_interleave(2 * closest_power_of_2)[0::2][: n - closest_power_of_2]
        )

def _fill_with_neg_inf(t):
    """FP16-compatible function that fills a tensor with -inf."""
    return t.float().fill_(float("-inf")).type_as(t)

def _gen_alibi_mask(n_head, max_pos):
    slopes = torch.Tensor(_get_interleave(n_head))
    position_point = torch.arange(max_pos) - max_pos + 1
    position_point = position_point.unsqueeze(0).unsqueeze(0).expand(n_head, -1, -1)
    diag = torch.diag(position_point[0])
    position_point = position_point - diag.unsqueeze(0).unsqueeze(0).transpose(-1, -2)
    alibi = slopes.unsqueeze(1).unsqueeze(1) * position_point
    alibi = alibi.view(n_head, 1, max_pos)
    alibi_mask = torch.triu(_fill_with_neg_inf(torch.zeros([max_pos, max_pos])), 1)
    alibi_mask = alibi_mask.unsqueeze(0) + alibi
    return alibi_mask


class EETBaichuanSelfMaskedAttention():
    def __init__(self, config, model_dict, layer_id, data_type=torch.float32, is_int8=False):
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

    @staticmethod
    def from_torch(config, model_dict, layer_id, data_type=torch.float32, is_int8=False):
        attention = EETBaichuanSelfMaskedAttention(config, model_dict, layer_id, data_type=data_type, is_int8=is_int8)
        return attention

 
class EETGatedFeedForward():
    def __init__(self, config, model_dict, layer_id, data_type=torch.float32, name="out_cache", is_int8=False):
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

    @staticmethod
    def from_torch(config, model_dict, layer_id, data_type=torch.float32, name="out_cache", is_int8=False):
        feedforward = EETGatedFeedForward(config, model_dict, layer_id, data_type=data_type, name=name, is_int8=is_int8)
        return feedforward


class EETBaichuanDecoderLayer():
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
        alibi_mask=None,
    ):

        ''' self_attn -> ffn'''
        self_attn_out = self.attention(
            hidden_states=hidden_states,
            pre_padding_len=pre_padding_len,
            reorder_state=reorder_state,
            pre_layernorm=normalize_before,
            add_residual=add_residual,
            first_pass=first_pass, 
            relative_attention_bias=alibi_mask,
        )

        out = self.feedforward(
            self_attn_out,
            pre_layernorm=normalize_before,
            add_residual=add_residual
        )
        return out

    @staticmethod
    def from_torch(config, model_dict, layer_id, data_type=torch.float32, is_int8=False):
        attention = EETBaichuanSelfMaskedAttention.from_torch(config, model_dict, layer_id, data_type=data_type, is_int8=is_int8)
        feedforward = EETGatedFeedForward.from_torch(config, model_dict, layer_id, data_type=data_type, name="decoder_out_cache", is_int8=is_int8)

        layer = EETBaichuanDecoderLayer(config, attention, feedforward)
        return layer


class EETBaichuanDecoder():
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
        alibi_mask=None,
    ):
        for layer in self.layers:
                hidden_states = layer(
                hidden_states,
                position_ids=None,
                first_pass=first_pass,
                pre_padding_len=pre_padding_len,
                reorder_state=reorder_state,
                normalize_before=normalize_before,
                add_residual=True,
                alibi_mask=alibi_mask,
            )
        return hidden_states

    @staticmethod
    def from_torch(config, layer_model_dict, layer_num, data_type=torch.float32, is_int8=False):
        """from torch."""
        DecoderLayers = []
        for i in tqdm(range(layer_num), desc="[EET][INFO] loading weight..."):
            DecoderLayers.extend(
                [
                    EETBaichuanDecoderLayer.from_torch(config, layer_model_dict['layer.' + str(i)], i, data_type=data_type, is_int8=is_int8)
                ]
            )

        eet_decoder = EETBaichuanDecoder(DecoderLayers)
        return eet_decoder


class EETBaichuanModel():
    def __init__(self, config, decoder, layer_norm, embedding=None, max_batch=1, max_prompt_len=1024, max_full_seq_len=1024):
        self.embedding = embedding
        self.decoder = decoder
        self.layer_norm = layer_norm
        self.pre_padding_len = torch.empty(0).long()
        self.reorder_state = torch.empty(0).long()
        self.current_len = 0
        self.config = config
        self.max_batch = max_batch
        self.max_prompt_len = max_prompt_len
        self.past_kv_length = 0
        # baichuan alibi mask
        self.n_head = config.num_attention_heads
        self.max_cache_pos = max_full_seq_len
        self.first_run = True
        self.alibi_mask = None
        self.future_mask = torch.zeros(self.n_head, self.max_cache_pos, self.max_cache_pos)

    def get_alibi_mask(self, tensor, seq_length_with_past):
        if self.first_run:
            self.first_run = False
            self.future_mask = _gen_alibi_mask(self.n_head, self.max_cache_pos).to(tensor)
        if seq_length_with_past > self.max_cache_pos:
            self.max_cache_pos = seq_length_with_past
            self.future_mask = _gen_alibi_mask(self.n_head, self.max_cache_pos).to(tensor)

        mask = self.future_mask[
            :self.n_head, :seq_length_with_past, :seq_length_with_past
        ]
        return mask

    def __call__(
        self,
        input_ids,
        first_pass=True,
        position_ids=None,
        token_type_ids=None,
        attention_mask=None,
        reorder_state=None,
    ):
        if first_pass:
            self.past_kv_length = 0
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

        # baichuan alibi mask
        self.past_kv_length += input_shape[1]
        alibi_mask = self.get_alibi_mask(embedding_out, self.past_kv_length)
        if not first_pass:
            alibi_mask = alibi_mask[:, -1:, :].contiguous()
        else:
            alibi_mask = alibi_mask.contiguous()

        decoder_out = self.decoder(
            embedding_out,
            position_ids=position_ids,
            first_pass=first_pass,
            pre_padding_len=pre_padding_len,
            reorder_state=self.reorder_state,
            normalize_before=True,
            alibi_mask=alibi_mask,
        )

        decoder_out = self.layer_norm(decoder_out)
        return decoder_out

    @staticmethod
    def from_torch(baichuan_dict, cfg, max_batch, max_prompt_seq_len, max_full_seq_len, data_type=torch.float16, device_id=0):
        """from torch."""
        torch.set_grad_enabled(False)
        embedding_dict = {}
        layernorm_dict = {}
        decoder_dict = {}  
        is_int8 = True if data_type == torch.int8 else False
        data_type = torch.float16 if data_type == torch.int8 else data_type

        for k, v in baichuan_dict.items():
            if 'embed_tokens.' in k:
                k = k[k.find('weight'):]
                embedding_dict[k] = v
            if 'layer.' in k:
                decoder_dict[k] = v
            if 'norm.' in k:
                k = k[k.find('norm.'):]
                layernorm_dict[k] = v

        from itertools import groupby

        layer_model_dict = {k: dict(v) for k, v in groupby(list(decoder_dict.items()),
                                                           lambda item: item[0][:(item[0].index('.', item[0].index('.')+1))])}

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
        decoder = EETBaichuanDecoder.from_torch(meta_des, layer_model_dict, layer_num=cfg.num_hidden_layers, data_type=data_type, is_int8=is_int8)
        layer_norm = EETLayerNorm.from_torch(meta_des, layernorm_dict['norm.weight'], None, data_type)

        eet_model = EETBaichuanModel(cfg, decoder, layer_norm=layer_norm, embedding=embedding, max_batch=max_batch, max_prompt_len=max_prompt_seq_len, max_full_seq_len=max_full_seq_len)
        return eet_model


class EETBaichuanForCausalLM(GenerationMixin_EET):
    def __init__(self, config, baichuanmodel, lm_head=None):
        self.config = config
        self.model = baichuanmodel
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
            softmax_normalizer = shift_logits.max(-1).values ** 2
            z_loss = self.config.z_loss_weight * softmax_normalizer.mean()
            # Enable model parallelism
            shift_labels = shift_labels.to(shift_logits.device)
            loss = loss_fct(shift_logits, shift_labels) + z_loss

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

        model_inputs = {"input_ids": input_ids}

        model_inputs.update(
            {
                "attention_mask": attention_mask,
            }
        )
        return model_inputs


    @staticmethod
    def from_pretrained(pt_or_path, config, max_batch, max_prompt_seq_len, max_full_seq_len, data_type=torch.float32, device_id=0, model_attr="model"):
        """from pretrained."""
        torch.set_grad_enabled(False)
        # cfg = torch_model.config

        model_dict = {}
        baichuan_dict = {}
        lm_head_dict = {}
        if data_type == torch.int8:
            with open(pt_or_path, "rb") as f:
                model_dict = torch.load(f, map_location=torch.device("cpu"))
        else:
            ts_model = AutoModelForCausalLM.from_pretrained(pt_or_path, trust_remote_code=True, torch_dtype=data_type)
            model_dict = convert_baichuan_weights(ts_model.state_dict(), data_type=data_type)

        for k, v in model_dict.items():
            if 'lm_head' in k:
                k = k[k.find('weight'):]
                lm_head_dict[k] = v
            else:
                baichuan_dict[k] = v

        baichuanmodel = EETBaichuanModel.from_torch(baichuan_dict, config, max_batch=max_batch, max_prompt_seq_len=max_prompt_seq_len,
                                                   max_full_seq_len=max_full_seq_len, data_type=data_type)


        lm_head = nn.Linear(config.hidden_size, config.vocab_size, bias=False)
        lm_head.load_state_dict(lm_head_dict)
        lm_head = lm_head.half().cuda()
        eet_model = EETBaichuanForCausalLM(config, baichuanmodel, lm_head)

        return eet_model


def convert_baichuan_weights(model_dict, data_type=torch.int8, model_attr="model"):
    baichuan_dict = {}
    prefix_len = len(model_attr)

    if data_type == torch.int8:
        for k, v in tqdm(model_dict.items(), desc="[EET][INFO] model weight preprocessing..."):
            k = k[prefix_len+1:] if model_attr in k else k
            if 'layers.' in k:
                k = convert_name(k, "baichuan")
                k = k[k.find('layer.'):]
                if 'layernorm' in k:
                    baichuan_dict[k] = v
                else:
                    unprocessed_weights = v.transpose(0, 1).contiguous()
                    processed_weights, processed_scale = quant_and_preprocess_weights(unprocessed_weights, torch.int8, False)
                    baichuan_dict[k] = processed_weights                
                    baichuan_dict[k.replace('.weight', '.scale')] = processed_scale
            else:
                baichuan_dict[k] = v
    else:
        for k, v in tqdm(model_dict.items(), desc="[EET][INFO] model weight preprocessing..."):
            k = k[prefix_len+1:] if model_attr in k else k
            if 'layers.' in k:
                k = convert_name(k, "baichuan")
                k = k[k.find('layer.'):]

            baichuan_dict[k] = v.to(data_type)

    return baichuan_dict

