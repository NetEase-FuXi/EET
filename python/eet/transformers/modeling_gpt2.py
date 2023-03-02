#
# Created by djz on 2021/01/21.
#
"""EET transformers gpt2 model. """

from distutils.command.config import config
import math
import time
import torch
import torch.nn as nn
from torch import Tensor
from typing import Any, Dict, List, Optional, Tuple
from transformers import GPT2Model,GPT2LMHeadModel,GPT2DoubleHeadsModel,GPT2ForSequenceClassification,GPT2ForTokenClassification
from transformers.file_utils import ModelOutput
from EET import MetaDesc as meta_desc
from EET import LayerNorm as eet_layernorm
from EET import FeedForwardNetwork as eet_ffn
from EET import Embedding as eet_embedding
from EET import MaskedMultiHeadAttention as eet_attention
from EET import CrossMultiHeadAttention as eet_cross_attention
from ..pipelines.generation import GenerationMixin_EET
import logging
logger = logging.getLogger(__name__)

BEGIN_OF_PARAM = 12

__all__ = [
    'EETLayerNorm', 'EETGPT2Embedding', 'EETGPT2Feedforward', 'EETGPT2Attention', 'EETGPT2DecoderLayer', 'EETGPT2Decoder', 'EETGPT2Model',
    'EETGPT2LMHeadModel','EETGPT2DoubleHeadsModel','EETGPT2ForSequenceClassification','EETGPT2ForTokenClassification'
]
class CausalLMOutputWithCrossAttentions(ModelOutput):
    loss: Optional[torch.FloatTensor] = None
    logits: torch.FloatTensor = None
    past_key_values: Optional[Tuple[Tuple[torch.FloatTensor]]] = None
    hidden_states: Optional[Tuple[torch.FloatTensor]] = None
    attentions: Optional[Tuple[torch.FloatTensor]] = None
    cross_attentions: Optional[Tuple[torch.FloatTensor]] = None

class GPT2DoubleHeadsModelOutput(ModelOutput):
    loss: Optional[torch.FloatTensor] = None
    mc_loss: Optional[torch.FloatTensor] = None
    logits: torch.FloatTensor = None
    mc_logits: torch.FloatTensor = None
    past_key_values: Optional[Tuple[Tuple[torch.FloatTensor]]] = None
    hidden_states: Optional[Tuple[torch.FloatTensor]] = None
    attentions: Optional[Tuple[torch.FloatTensor]] = None

class SequenceClassifierOutputWithPast(ModelOutput):
    loss: Optional[torch.FloatTensor] = None
    logits: torch.FloatTensor = None
    past_key_values: Optional[Tuple[Tuple[torch.FloatTensor]]] = None
    hidden_states: Optional[Tuple[torch.FloatTensor]] = None
    attentions: Optional[Tuple[torch.FloatTensor]] = None

class TokenClassifierOutput(ModelOutput):
    loss: Optional[torch.FloatTensor] = None
    logits: torch.FloatTensor = None
    hidden_states: Optional[Tuple[torch.FloatTensor]] = None
    attentions: Optional[Tuple[torch.FloatTensor]] = None


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
        self.embedding = eet_embedding(meta_des,self.embedding_weights,self.position_weights,self.token_type_weights,self.Layernorm_weights,self.Layernorm_bias, 'emb_cache')
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
        self.ffn = eet_ffn(meta_des,self.intermediate_weights,self.intermediate_bias,self.output_weights,self.output_bias,self.layernorm_weights,self.layernorm_bias, 'ffn_out_cache')
    def __call__(self,
                input_id,
                pre_layernorm = True,
                add_residual = True):
        return self.ffn.forward(input_id,pre_layernorm,add_residual)
    
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
            self.qkv_weights = torch.t([x[1] for x in model_dict.items() if 'c_attn.weight' in x[0]][0].transpose(0,1)).contiguous().cuda().type(data_type)
            self.q_bias = [x[1] for x in model_dict.items() if 'c_attn.bias' in x[0]][0][:emb_size].cuda().type(data_type)
            self.k_bias = [x[1] for x in model_dict.items() if 'c_attn.bias' in x[0]][0][emb_size:emb_size*2].cuda().type(data_type)
            self.v_bias = [x[1] for x in model_dict.items() if 'c_attn.bias' in x[0]][0][emb_size*2:].cuda().type(data_type)
            self.out_weights = [x[1] for x in model_dict.items() if 'attn.c_proj.weight' in x[0]][0].cuda().type(data_type)
            self.out_bias = [x[1] for x in model_dict.items() if 'attn.c_proj.bias' in x[0]][0].cuda().type(data_type)
            self.attention = eet_attention(meta_des,self.qkv_weights,self.q_bias,self.k_bias,self.v_bias,self.out_weights,self.out_bias,self.layernorm_weights,self.layernorm_bias)

    def __call__(self,
                input_id,
                pre_padding_len,
                reorder_state = None,
                encoder_out = None,
                encoder_padding_mask  = None,
                pre_layernorm = False,
                add_residual = True,
                first_pass = False):
                
        if self.cross_attn:
            return self.attention.forward(input_id,encoder_out,pre_padding_len,pre_layernorm,add_residual,encoder_padding_mask,first_pass)
        else:
            return self.attention.forward(input_id,pre_padding_len,reorder_state,pre_layernorm,add_residual, first_pass)

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
        self.add_residual = True
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
                        add_residual = self.add_residual,
                        first_pass = first_pass)

            cross_attn_out = self.cross_attention(input_id = self_attn_out,
                        pre_padding_len = pre_padding_len,
                        encoder_out = encoder_out,
                        encoder_padding_mask = encoder_attention_mask,
                        pre_layernorm = self.normalize_before,
                        add_residual = self.add_residual,
                        first_pass = first_pass)

            out = self.feedforward(cross_attn_out,
                        pre_layernorm = self.normalize_before,
                        add_residual = self.add_residual)
        else:
            self_attn_out = self.attetion(input_id = x,
                        pre_padding_len = pre_padding_len,
                        reorder_state = reorder_state,
                        pre_layernorm = self.normalize_before,
                        add_residual = self.add_residual,
                        first_pass = first_pass)

            out = self.feedforward(self_attn_out,
                        pre_layernorm = self.normalize_before,
                        add_residual = self.add_residual)          

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
        self.current_len = 0
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
        input_shape = input_ids.size()
        batch_size = input_shape[0]
        # Attention mask.
        if attention_mask is  None:
            pre_padding_len = self.self_attn_padding_mask
        else:
            pre_padding_len = torch.sum(1 - attention_mask, 1, True).cuda().long()
        if position_ids is None:
            if first_pass is True:
                self.current_len = input_shape[1]
                position_ids = self.position_ids[:, :self.current_len]
            else:
                self.current_len = self.current_len + 1
                position_ids = self.position_ids[:, self.current_len-1:self.current_len]

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
        """from_pretrained."""
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
        meta_des = meta_desc(batch_size, cfg.n_head, cfg.n_embd, 0, 0, cfg.n_layer , cfg.n_positions,full_seq_len, data_type, device, False, activation_fn)
        layer_norm = EETLayerNorm.from_torch(meta_des,layernorm_dict['ln_f.weight'],layernorm_dict['ln_f.bias'],data_type)
        embedding = EETGPT2Embedding.from_torch(meta_des,embedding_dict,data_type)
        # embedding = None
        decoder = EETGPT2Decoder.from_torch(layer_model_dict,meta_des, cfg,data_type)
        eet_model =  EETGPT2Model(cfg,embedding, decoder,layer_norm)
        return eet_model

    @staticmethod
    def from_torch(torch_model,max_batch,full_seq_len,data_type = torch.float32):
        """from torch."""
        torch.set_grad_enabled(False)
        model_dict = {}
        embedding_dict = {}
        layernorm_dict = {}

        cfg = torch_model.config

        for k, v in torch_model.state_dict().items():
            k = k[BEGIN_OF_PARAM:]
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
        meta_des = meta_desc(batch_size, cfg.n_head, cfg.n_embd, 0, 0, cfg.n_layer, cfg.n_positions,full_seq_len, data_type, device, False, activation_fn)
        layer_norm = EETLayerNorm.from_torch(meta_des,layernorm_dict['ln_f.weight'],layernorm_dict['ln_f.bias'],data_type)
        embedding = EETGPT2Embedding.from_torch(meta_des,embedding_dict,data_type)
        # embedding = None
        decoder = EETGPT2Decoder.from_torch(layer_model_dict,meta_des, cfg,data_type)
        eet_model =  EETGPT2Model(cfg,embedding, decoder,layer_norm)
        return eet_model

class EETGPT2LMHeadModel(GenerationMixin_EET):
    def __init__(self, gpt2model,lm_head,config):
        self.transformer = gpt2model
        self.lm_head = lm_head
        self.config = config
        self.main_input_name = "input_ids"
        self.device = "cuda:0"

    def prepare_inputs_for_generation(self, input_ids,first_pass = True, past=None, **kwargs):
        token_type_ids = kwargs.get("token_type_ids", None)
        # only last token for inputs_ids if past is defined in kwargs
        if first_pass == False:
            input_ids = input_ids[:, -1].unsqueeze(-1)
            if token_type_ids is not None:
                token_type_ids = token_type_ids[:, -1].unsqueeze(-1)

        attention_mask = kwargs.get("attention_mask", None)
        position_ids = kwargs.get("position_ids", None)

        if attention_mask is not None and position_ids is None:
            # create position_ids on the fly for batch generation
            position_ids = attention_mask.long().cumsum(-1) - 1
            position_ids.masked_fill_(attention_mask == 0, 1)
            if first_pass == False:
                position_ids = position_ids[:, -1].unsqueeze(-1)
            position_ids = position_ids.contiguous()
        else:
            position_ids = None
        input_ids = input_ids.contiguous()

        return {
            "input_ids": input_ids,
            "past_key_values": past,
            "use_cache": kwargs.get("use_cache"),
            "position_ids": position_ids,
            "attention_mask": attention_mask,
            "token_type_ids": token_type_ids,
        }

    def forward(
        self,
        input_ids=None,
        past_key_values=None,
        attention_mask=None,
        token_type_ids=None,
        position_ids=None,
        head_mask=None,
        inputs_embeds=None,
        encoder_hidden_states=None,
        encoder_attention_mask=None,
        labels=None,
        use_cache=None,
        output_attentions=None,
        output_hidden_states=None,
        return_dict=None,
        first_pass = True,
    ):
        transformer_outputs = self.transformer(
            input_ids = input_ids,
            encoder_out = None,
            first_pass = first_pass,
            position_ids = position_ids,
            token_type_ids = token_type_ids,
            attention_mask = attention_mask,
            reorder_state = None,
        )


        lm_logits = self.lm_head(transformer_outputs)

        loss = None

        return CausalLMOutputWithCrossAttentions(
            loss=loss,
            logits=lm_logits,
            past_key_values=None,
            hidden_states=None,
            attentions=None,
            cross_attentions=None,
        )

    def __call__(
        self,
        input_ids=None,
        past_key_values=None,
        attention_mask=None,
        token_type_ids=None,
        position_ids=None,
        head_mask=None,
        inputs_embeds=None,
        labels=None,
        use_cache=None,
        output_attentions=None,
        output_hidden_states=None,
        return_dict=None,
        first_pass = True,
    ):
        return self.forward(        
            input_ids=input_ids,
            past_key_values=past_key_values,
            attention_mask=attention_mask,
            token_type_ids=token_type_ids,
            position_ids=position_ids,
            head_mask=head_mask,
            inputs_embeds=inputs_embeds,
            labels=None,
            use_cache=use_cache,
            output_attentions=output_attentions,
            output_hidden_states=output_hidden_states,
            return_dict=return_dict,
            first_pass = first_pass,
            )

    def from_pretrained(model_id_or_path: str,max_batch, full_seq_len,data_type):
        """from_pretrained"""
        torch.set_grad_enabled(False)
        model_dict = {}
        embedding_dict = {}
        torch_model = GPT2LMHeadModel.from_pretrained(model_id_or_path)
        if data_type == torch.float32:
            torch_model = torch_model.float()
        else:
            torch_model = torch_model.half()

        gpt2 = EETGPT2Model.from_torch(torch_model,max_batch,full_seq_len,data_type)

        lm_head = torch_model.lm_head.cuda()
        model =  EETGPT2LMHeadModel(gpt2, lm_head,torch_model.config)

        return model

class EETGPT2DoubleHeadsModel(GenerationMixin_EET):
    def __init__(self, gpt2model,lm_head,multiple_choice_head,config):
        self.transformer = gpt2model
        self.lm_head = lm_head
        self.multiple_choice_head =multiple_choice_head
        self.config = config
        self.device = "cuda:0"

    def prepare_inputs_for_generation(self, input_ids,first_pass = True, past=None, **kwargs):
        token_type_ids = kwargs.get("token_type_ids", None)

        # only last token for inputs_ids if past is defined in kwargs
        if first_pass == False:
            input_ids = input_ids[:, -1].unsqueeze(-1)
            if token_type_ids is not None:
                token_type_ids = token_type_ids[:, -1].unsqueeze(-1)

        attention_mask = kwargs.get("attention_mask", None)
        position_ids = kwargs.get("position_ids", None)

        if attention_mask is not None and position_ids is None:
            # create position_ids on the fly for batch generation
            position_ids = attention_mask.long().cumsum(-1) - 1
            position_ids.masked_fill_(attention_mask == 0, 1)
            if first_pass == False:
                position_ids = position_ids[:, -1].unsqueeze(-1)
            position_ids = position_ids.contiguous()

        else:
            position_ids = None
        input_ids = input_ids.contiguous()

        return {
            "input_ids": input_ids,
            "past_key_values": past,
            "use_cache": kwargs.get("use_cache"),
            "position_ids": position_ids,
            "attention_mask": attention_mask,
            "token_type_ids": token_type_ids,
        }


    def forward(
        self,
        input_ids=None,
        past_key_values=None,
        attention_mask=None,
        token_type_ids=None,
        position_ids=None,
        head_mask=None,
        inputs_embeds=None,
        mc_token_ids=None,
        labels=None,
        mc_labels=None,
        use_cache=None,
        output_attentions=None,
        output_hidden_states=None,
        return_dict=None,
        first_pass = True,
        **kwargs,
    ):
        transformer_outputs = self.transformer(
            self,
            input_ids = input_ids,
            encoder_out = None,
            first_pass = first_pass,
            position_ids = position_ids,
            token_type_ids = token_type_ids,
            attention_mask = attention_mask,
            reorder_state = None,
        )


        hidden_states = transformer_outputs

        lm_logits = self.lm_head(hidden_states)
        mc_logits = self.multiple_choice_head(hidden_states, mc_token_ids).squeeze(-1)

        mc_loss = None

        lm_loss = None

        return GPT2DoubleHeadsModelOutput(
            loss=lm_loss,
            mc_loss=mc_loss,
            logits=lm_logits,
            mc_logits=mc_logits,
            past_key_values=None,
            hidden_states=None,
            attentions=None,
        )

    @staticmethod
    def _reorder_cache(past: Tuple[Tuple[torch.Tensor]], beam_idx: torch.Tensor) -> Tuple[Tuple[torch.Tensor]]:
        """
        This function is used to re-order the `past_key_values` cache if [`~PreTrainedModel.beam_search`] or
        [`~PreTrainedModel.beam_sample`] is called. This is required to match `past_key_values` with the correct
        beam_idx at every generation step.
        """
        return tuple(
            tuple(past_state.index_select(0, beam_idx.to(past_state.device)) for past_state in layer_past)
            for layer_past in past
        )

    def from_pretrained(model_id_or_path: str,max_batch,full_seq_len, data_type):
        """from_pretrained"""
        torch.set_grad_enabled(False)
        model_dict = {}
        embedding_dict = {}
        torch_model = GPT2LMHeadModel.from_pretrained(model_id_or_path)
        if data_type == torch.float32:
            torch_model = torch_model.float()
        else:
            torch_model = torch_model.half()

        gpt2 = EETGPT2Model.from_torch(torch_model,max_batch,full_seq_len,data_type)

        lm_head = torch_model.lm_head.cuda()
        multiple_choice_head = torch_model.multiple_choice_head.cuda()

        model =  EETGPT2DoubleHeadsModel(gpt2, lm_head,multiple_choice_head,torch_model.config)

        return model

class EETGPT2ForSequenceClassification(GenerationMixin_EET):
    def __init__(self, gpt2model,score,config):
        self.config = config
        self.transformer = gpt2model
        self.score = score
        self.device = "cuda:0"

    def forward(
        self,
        input_ids=None,
        past_key_values=None,
        attention_mask=None,
        token_type_ids=None,
        position_ids=None,
        head_mask=None,
        inputs_embeds=None,
        labels=None,
        use_cache=None,
        output_attentions=None,
        output_hidden_states=None,
        return_dict=None,
        first_pass = True,
    ):
        transformer_outputs = self.transformer(
            self,
            input_ids = input_ids,
            encoder_out = None,
            first_pass = first_pass,
            position_ids = position_ids,
            token_type_ids = token_type_ids,
            attention_mask = attention_mask,
            reorder_state = None,
        )


        logits = self.score(transformer_outputs)
        if input_ids is not None:
            batch_size, sequence_length = input_ids.shape[:2]
        else:
            batch_size, sequence_length = inputs_embeds.shape[:2]

        assert (
            self.config.pad_token_id is not None or batch_size == 1
        ), "Cannot handle batch sizes > 1 if no padding token is defined."
        if self.config.pad_token_id is None:
            sequence_lengths = -1
        else:
            if input_ids is not None:
                sequence_lengths = torch.ne(input_ids, self.config.pad_token_id).sum(-1) - 1
            else:
                sequence_lengths = -1
                logger.warning(
                    f"{self.__class__.__name__} will not detect padding tokens in `inputs_embeds`. Results may be "
                    f"unexpected if using padding tokens in conjunction with `inputs_embeds.`"
                )

        pooled_logits = logits[torch.arange(batch_size, device=self.device), sequence_lengths]
        loss = None

        return SequenceClassifierOutputWithPast(
            loss=loss,
            logits=pooled_logits,
            past_key_values=None,
            hidden_states=None,
            attentions=None,
        )

    def from_pretrained(model_id_or_path: str,max_batch, full_seq_len,data_type):
        """from_pretrained"""
        torch.set_grad_enabled(False)
        model_dict = {}
        embedding_dict = {}
        torch_model = GPT2LMHeadModel.from_pretrained(model_id_or_path)
        if data_type == torch.float32:
            torch_model = torch_model.float()
        else:
            torch_model = torch_model.half()

        gpt2 = EETGPT2Model.from_torch(torch_model,max_batch,full_seq_len, data_type)

        lm_head = torch_model.lm_head.cuda()
        multiple_choice_head = torch_model.multiple_choice_head.cuda()

        config = torch_model.config
        model =  EETGPT2ForSequenceClassification(gpt2, lm_head,multiple_choice_head,config)
        return model

class EETGPT2ForTokenClassification(GenerationMixin_EET):
    def __init__(self, gpt2model,classifier,config):
        self.transformer = gpt2model
        self.classifier = classifier
        self.config - config
        self.device = "cuda:0"

    def forward(
        self,
        input_ids=None,
        past_key_values=None,
        attention_mask=None,
        token_type_ids=None,
        position_ids=None,
        head_mask=None,
        inputs_embeds=None,
        labels=None,
        use_cache=None,
        output_attentions=None,
        output_hidden_states=None,
        return_dict=None,
        first_pass = True,
    ):
        transformer_outputs = self.transformer(
            self,
            input_ids = input_ids,
            encoder_out = None,
            first_pass = first_pass,
            position_ids = position_ids,
            token_type_ids = token_type_ids,
            attention_mask = attention_mask,
            reorder_state = None,
        )


        logits = self.classifier(transformer_outputs)
        
        loss = None

        return TokenClassifierOutput(
            loss=loss,
            logits=logits,
            hidden_states=None,
            attentions=None,
        )


    def from_pretrained(model_id_or_path: str,max_batch,full_seq_len, data_type):
        """from_pretrained"""
        torch.set_grad_enabled(False)
        model_dict = {}
        embedding_dict = {}
        torch_model = GPT2LMHeadModel.from_pretrained(model_id_or_path)
        if data_type == torch.float32:
            torch_model = torch_model.float()
        else:
            torch_model = torch_model.half()

        gpt2 = EETGPT2Model.from_torch(torch_model,max_batch,full_seq_len,data_type)

        classifier = torch_model.classifier.cuda()

        model =  EETGPT2ForTokenClassification(gpt2,classifier,torch_model.config)

        return model
