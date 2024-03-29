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

from eet.transformers.encoder_decoder import *
from eet.utils.mapping import convert_name
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
    'EETLayerNorm', 'EETGPT2Embedding', 'EETGPT2Model',
    'EETGPT2LMHeadModel', 'EETGPT2DoubleHeadsModel', 'EETGPT2ForSequenceClassification', 'EETGPT2ForTokenClassification'
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
    def __init__(self, meta_des, layernorm_weights, layernorm_bias, data_type=torch.float32):
        self.layernorm_weights = layernorm_weights.cuda().type(data_type)
        self.layernorm_bias = layernorm_bias.cuda().type(data_type)
        self.layernorm = eet_layernorm(meta_des, self.layernorm_weights, self.layernorm_bias)

    def __call__(self,
                 input_ids):
        return self.layernorm.layer_norm(input_ids)

    @staticmethod
    def from_torch(meta_des, layernorm_weights, layernorm_bias, data_type=torch.float32):
        layernorm = EETLayerNorm(meta_des, layernorm_weights, layernorm_bias, data_type=data_type)
        return layernorm


class EETGPT2Embedding():
    def __init__(self, meta_des, embedding_dict, data_type=torch.float32):
        self.embedding_weights = embedding_dict['wte.weight'].cuda().type(data_type)
        self.position_weights = embedding_dict['wpe.weight'].cuda().type(data_type)
        # not use token_type
        self.token_type_ids = torch.empty(0).long()
        self.token_type_weights = torch.empty(0)
        self.if_layernorm = False
        # not use layernorm
        self.Layernorm_weights = torch.empty(0)
        self.Layernorm_bias = torch.empty(0)
        self.embedding = eet_embedding(meta_des, self.embedding_weights, self.position_weights, self.token_type_weights, self.Layernorm_weights, self.Layernorm_bias, 'emb_cache')

    def __call__(self,
                 input_ids,
                 position_ids,
                 token_type_ids):
        if_layernorm = False
        if token_type_ids is None:
            token_type_ids = self.token_type_ids
        return self.embedding.forward_transformers(input_ids, position_ids, token_type_ids, if_layernorm)

    @staticmethod
    def from_torch(meta_des, embedding_dict, data_type=torch.float32):
        embedding = EETGPT2Embedding(meta_des, embedding_dict, data_type=data_type)
        return embedding


class EETGPT2Model():
    def __init__(self, config, embedding, decoder, layer_norm):
        self.embedding = embedding
        self.decoder = decoder
        self.layer_norm = layer_norm
        self.position_ids = torch.arange(0, config.n_positions).reshape(1, config.n_positions).cuda()
        self.self_attn_padding_mask = torch.empty(0).long()
        self.encoder_attention_mask = torch.empty(0)
        self.reorder_state = torch.empty(0).long()
        self.current_len = 0

    def __call__(
        self,
        input_ids,
        encoder_out=None,
        first_pass=True,
        position_ids=None,
        token_type_ids=None,
        attention_mask=None,
        reorder_state=None,
    ):
        input_shape = input_ids.size()
        # Attention mask.
        if attention_mask is None:
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
        embedding_out = self.embedding(input_ids, position_ids, token_type_ids)

        if reorder_state is not None:
            self.reorder_state = reorder_state.long()

        decoder_out = self.decoder(
            embedding_out,
            encoder_outputs=encoder_out,
            first_pass=first_pass,
            pre_padding_len=pre_padding_len,
            head_mask=None,
            reorder_state=self.reorder_state,
            normalize_before=True,
        )

        decoder_out = self.layer_norm(decoder_out)
        return decoder_out

    @staticmethod
    def from_pretrained(model_id_or_path: str, max_batch, full_seq_len, data_type=torch.float32):
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
                k = convert_name(k, "gpt2")
                k = k[k.find('layer.'):]
                model_dict[k] = v
            if 'ln_f' in k:
                layernorm_dict[k] = v

        from itertools import groupby

        layer_model_dict = {k: dict(v) for k, v in groupby(list(model_dict.items()),
                                                           lambda item: item[0][:(item[0].index('.', item[0].index('.')+1))])}

        device = "cuda:0"
        activation_fn = cfg.activation_function
        batch_size = max_batch
        full_seq_len = full_seq_len
        meta_des = meta_desc(dtype=data_type,
                             batch_size=batch_size,
                             head_num=cfg.n_head,
                             hidden_units=cfg.n_embd,
                             layer_num=cfg.n_layer,
                             max_seq_len=cfg.n_positions,
                             max_full_seq_len=full_seq_len,
                             activation_fn=activation_fn,
                             cuda_device=device)
        layer_norm = EETLayerNorm.from_torch(meta_des, layernorm_dict['ln_f.weight'], layernorm_dict['ln_f.bias'], data_type)
        embedding = EETGPT2Embedding.from_torch(meta_des, embedding_dict, data_type)
        # embedding = None
        decoder = EETDecoder.from_torch(meta_des, layer_model_dict, layer_num=cfg.n_layer, data_type=data_type, add_cross_attn=False, bias=True, is_standard=False)
        eet_model = EETGPT2Model(cfg, embedding, decoder, layer_norm)
        return eet_model

    @staticmethod
    def from_torch(torch_model, max_batch, full_seq_len, data_type=torch.float32):
        """from torch."""
        torch.set_grad_enabled(False)
        model_dict = {}
        embedding_dict = {}
        layernorm_dict = {}

        cfg = torch_model.config

        for k, v in torch_model.state_dict().items():
            if 'e.' in k:
                embedding_dict[k] = v
            if 'h.' in k:
                k = convert_name(k, "gpt2")
                k = k[k.find('layer.'):]
                model_dict[k] = v
            if 'ln_f' in k:
                layernorm_dict[k] = v

        from itertools import groupby

        layer_model_dict = {k: dict(v) for k, v in groupby(list(model_dict.items()),
                                                           lambda item: item[0][:(item[0].index('.', item[0].index('.')+1))])}

        device = "cuda:0"
        activation_fn = cfg.activation_function
        batch_size = max_batch
        full_seq_len = full_seq_len
        meta_des = meta_desc(dtype=data_type,
                             batch_size=batch_size,
                             head_num=cfg.n_head,
                             hidden_units=cfg.n_embd,
                             layer_num=cfg.n_layer,
                             max_seq_len=cfg.n_positions,
                             max_full_seq_len=full_seq_len,
                             activation_fn=activation_fn,
                             cuda_device=device)
        layer_norm = EETLayerNorm.from_torch(meta_des, layernorm_dict['ln_f.weight'], layernorm_dict['ln_f.bias'], data_type)
        embedding = EETGPT2Embedding.from_torch(meta_des, embedding_dict, data_type)
        # embedding = None
        decoder = EETDecoder.from_torch(meta_des, layer_model_dict, layer_num=cfg.n_layer, data_type=data_type, add_cross_attn=False, bias=True, is_standard=False)
        eet_model = EETGPT2Model(cfg, embedding, decoder, layer_norm)
        return eet_model


class EETGPT2LMHeadModel(GenerationMixin_EET):
    def __init__(self, gpt2model, lm_head, config):
        self.transformer = gpt2model
        self.lm_head = lm_head
        self.config = config
        self.main_input_name = "input_ids"
        self.device = "cuda:0"

    def prepare_inputs_for_generation(self, input_ids, first_pass=True, past=None, **kwargs):
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
        first_pass=True,
    ):
        transformer_outputs = self.transformer(
            input_ids=input_ids,
            encoder_out=None,
            first_pass=first_pass,
            position_ids=position_ids,
            token_type_ids=token_type_ids,
            attention_mask=attention_mask,
            reorder_state=None,
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
        first_pass=True,
        self_past_key_values_length=0,
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
            first_pass=first_pass,
        )

    def from_pretrained(model_id_or_path: str, max_batch, full_seq_len, data_type):
        """from_pretrained"""
        torch.set_grad_enabled(False)
        model_dict = {}
        embedding_dict = {}
        torch_model = GPT2LMHeadModel.from_pretrained(model_id_or_path).to(data_type)

        gpt2 = EETGPT2Model.from_torch(torch_model, max_batch, full_seq_len, data_type)

        lm_head = torch_model.lm_head.cuda()
        model = EETGPT2LMHeadModel(gpt2, lm_head, torch_model.config)

        return model


class EETGPT2DoubleHeadsModel(GenerationMixin_EET):
    def __init__(self, gpt2model, lm_head, multiple_choice_head, config):
        self.transformer = gpt2model
        self.lm_head = lm_head
        self.multiple_choice_head = multiple_choice_head
        self.config = config
        self.device = "cuda:0"

    def prepare_inputs_for_generation(self, input_ids, first_pass=True, past=None, **kwargs):
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
        first_pass=True,
        **kwargs,
    ):
        transformer_outputs = self.transformer(
            self,
            input_ids=input_ids,
            encoder_out=None,
            first_pass=first_pass,
            position_ids=position_ids,
            token_type_ids=token_type_ids,
            attention_mask=attention_mask,
            reorder_state=None,
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

    def from_pretrained(model_id_or_path: str, max_batch, full_seq_len, data_type):
        """from_pretrained"""
        torch.set_grad_enabled(False)
        model_dict = {}
        embedding_dict = {}
        torch_model = GPT2LMHeadModel.from_pretrained(model_id_or_path).to(data_type)

        gpt2 = EETGPT2Model.from_torch(torch_model, max_batch, full_seq_len, data_type)
        lm_head = torch_model.lm_head.cuda()
        multiple_choice_head = torch_model.multiple_choice_head.cuda()
        model = EETGPT2DoubleHeadsModel(gpt2, lm_head, multiple_choice_head, torch_model.config)

        return model


class EETGPT2ForSequenceClassification(GenerationMixin_EET):
    def __init__(self, gpt2model, score, config):
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
        first_pass=True,
        **kwargs,
    ):
        transformer_outputs = self.transformer(
            self,
            input_ids=input_ids,
            encoder_out=None,
            first_pass=first_pass,
            position_ids=position_ids,
            token_type_ids=token_type_ids,
            attention_mask=attention_mask,
            reorder_state=None,
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

    def from_pretrained(model_id_or_path: str, max_batch, full_seq_len, data_type):
        """from_pretrained"""
        torch.set_grad_enabled(False)
        model_dict = {}
        embedding_dict = {}
        torch_model = GPT2LMHeadModel.from_pretrained(model_id_or_path).to(data_type)

        gpt2 = EETGPT2Model.from_torch(torch_model,max_batch,full_seq_len, data_type)

        lm_head = torch_model.lm_head.cuda()
        multiple_choice_head = torch_model.multiple_choice_head.cuda()

        config = torch_model.config
        model =  EETGPT2ForSequenceClassification(gpt2, lm_head,multiple_choice_head,config)
        return model

class EETGPT2ForTokenClassification(GenerationMixin_EET):
    def __init__(self, gpt2model, classifier, config):
        self.transformer = gpt2model
        self.classifier = classifier
        self.config = config
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
        first_pass=True,
        **kwargs,
    ):
        transformer_outputs = self.transformer(
            self,
            input_ids=input_ids,
            encoder_out=None,
            first_pass=first_pass,
            position_ids=position_ids,
            token_type_ids=token_type_ids,
            attention_mask=attention_mask,
            reorder_state=None,
        )

        logits = self.classifier(transformer_outputs)
        loss = None

        return TokenClassifierOutput(
            loss=loss,
            logits=logits,
            hidden_states=None,
            attentions=None,
        )

    def from_pretrained(model_id_or_path: str, max_batch, full_seq_len, data_type):
        """from_pretrained"""
        torch.set_grad_enabled(False)
        model_dict = {}
        embedding_dict = {}
        torch_model = GPT2LMHeadModel.from_pretrained(model_id_or_path)
        torch_model.to(data_type)

        gpt2 = EETGPT2Model.from_torch(torch_model, max_batch, full_seq_len, data_type)
        classifier = torch_model.classifier.cuda()
        model = EETGPT2ForTokenClassification(gpt2, classifier, torch_model.config)

        return model
