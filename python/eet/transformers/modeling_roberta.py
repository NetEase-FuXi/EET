#
# Created by zsd on 2022/02/22.
#
"""EET transformers roberta model. """

import math
import time
import torch
import torch.nn as nn
import numpy as np
from torch import Tensor
from typing import Any, Dict, List, Optional, Tuple
from transformers import RobertaModel
from eet.transformers.encoder_decoder import *
from eet.transformers.modeling_bert import EETBertEmbedding
from eet.utils.mapping import convert_name
from transformers import  (
    RobertaModel,
    RobertaForCausalLM,
    RobertaForMaskedLM,
    RobertaForSequenceClassification,
    RobertaForMultipleChoice,
    RobertaForTokenClassification,
    RobertaForQuestionAnswering,
)
from transformers.modeling_outputs import (
    CausalLMOutputWithCrossAttentions,
    MaskedLMOutput,
    MultipleChoiceModelOutput,
    QuestionAnsweringModelOutput,
    SequenceClassifierOutput,
    TokenClassifierOutput,
)
from EET import MetaDesc as meta_desc
from EET import FeedForwardNetwork as eet_ffn
from EET import MultiHeadAttention as eet_attention
from EET import Embedding as eet_embedding


class EETRobertaModel():
    def __init__(self, config, embedding, encoder, pooler=None):
        self.embedding = embedding
        self.encoder = encoder
        self.pre_padding_len = torch.empty(0).long()
        self.padding_idx = config.pad_token_id
        if pooler is not None:
            pooler = pooler.cuda()
        self.pooler = pooler
    def __call__(
            self,
            input_ids,
            position_ids=None,
            token_type_ids=None,
            attention_mask=None,
    ):
        '''
        attention_mask:attention_padding_mask(:obj:`torch.FloatTensor` of shape :obj:`(batch_size, sequence_length)`, `optional`, defaults to :obj:`None`):
            Mask to avoid performing attention on the padding token indices of the encoder input.)
            Mask values selected in ``[0, 1]``:
            ``1`` for tokens that are NOT MASKED, ``0`` for MASKED tokens.
        '''
        input_shape = input_ids.size()

        # Same as BertEmbeddings with a tiny tweak for positional embeddings indexing
        mask = input_ids.ne(self.padding_idx).int()
        incremental_indices = (torch.cumsum(mask, dim=1).type_as(mask)) * mask
        position_ids = incremental_indices.long() + self.padding_idx

        if token_type_ids is None:
            token_type_ids = torch.zeros(input_shape, dtype=torch.long, device=input_ids.device)

        if attention_mask is None:
            pre_padding_len = self.pre_padding_len
        else:
            # transformers 0 - padding;1 - nopadding
            pre_padding_len = torch.sum(1 - attention_mask, 1).long().cuda()

        embedding_out = self.embedding(input_ids, position_ids, token_type_ids)

        sequence_output = self.encoder(embedding_out,
                                   pre_padding_len=pre_padding_len,
                                   normalize_before=False)

        pooled_output = self.pooler(sequence_output) if self.pooler is not None else None

        return sequence_output,pooled_output
    @staticmethod
    def from_pretrained(model_id_or_path: str, max_batch, data_type, device_id=0):
        """from_pretrained."""
        torch.set_grad_enabled(False)
        model_dict = {}
        embedding_dict = {}
        torch_model = RobertaModel.from_pretrained(model_id_or_path)
        cfg = torch_model.config
        model_name = cfg.model_type

        for k, v in torch_model.state_dict().items():
            if 'embeddings.' in k:
                embedding_dict[k] = v
            if 'layer.' in k:
                # Structure mapping
                k = convert_name(k, model_name)
                k = k[k.find('layer.'):]
                model_dict[k] = v

        # Group by layer id in layer_model_dict's keys
        from itertools import groupby
        layer_model_dict = {k: dict(v) for k, v in groupby(list(model_dict.items()),
                                                           lambda item: item[0][:(item[0].index('.', item[0].index('.')+1))])}

        device = "cpu" if device_id < 0 else f"cuda:{device_id}"
        activation_fn = cfg.hidden_act
        batch_size = max_batch
        config = meta_desc(batch_size, cfg.num_attention_heads, cfg.hidden_size, cfg.num_hidden_layers,
                           cfg.max_position_embeddings, cfg.max_position_embeddings, data_type, device, False,
                           activation_fn)

        embedding = EETBertEmbedding.from_torch(config, embedding_dict, data_type)
        # embedding = None
        encoder = EETEncoder.from_torch(config, layer_model_dict, cfg.num_hidden_layers, data_type)
        if data_type == torch.float32:
            torch_model = torch_model.float()
        else:
            torch_model = torch_model.half()
        eet_model = EETRobertaModel(cfg, embedding, encoder,torch_model.pooler)
        return eet_model

    def from_torch(torch_model, max_batch, data_type, device_id=0):
        """from torch."""
        torch.set_grad_enabled(False)
        model_dict = {}
        embedding_dict = {}
        cfg = torch_model.config
        model_name = cfg.model_type

        for k, v in torch_model.roberta.state_dict().items():
            if 'embeddings.' in k:
                embedding_dict[k] = v
            if 'layer.' in k:
                # Structure mapping
                k = convert_name(k, model_name)
                k = k[k.find('layer.'):]
                model_dict[k] = v

        # Group by layer id in layer_model_dict's keys
        from itertools import groupby
        layer_model_dict = {k: dict(v) for k, v in groupby(list(model_dict.items()),
                                                           lambda item: item[0][:(item[0].index('.', item[0].index('.')+1))])}

        device = "cpu" if device_id < 0 else f"cuda:{device_id}"
        activation_fn = cfg.hidden_act
        batch_size = max_batch
        config = meta_desc(batch_size, cfg.num_attention_heads, cfg.hidden_size, cfg.num_hidden_layers,
                           cfg.max_position_embeddings, cfg.max_position_embeddings, data_type, device, False,
                           activation_fn)

        embedding = EETBertEmbedding.from_torch(config, embedding_dict, data_type)
        # embedding = None
        encoder = EETEncoder.from_torch(config, layer_model_dict, cfg.num_hidden_layers, data_type)
        if data_type == torch.float32:
            torch_model = torch_model.float()
        else:
            torch_model = torch_model.half()
        eet_model = EETRobertaModel(cfg, embedding, encoder, torch_model.roberta.pooler)
        return eet_model


class EETRobertaForCausalLM():
    def __init__(self,roberta,lm_head,config):
        self.config = config
        self.roberta = roberta
        self.lm_head = lm_head

    def __call__(
        self,
        input_ids,
        position_ids = None,
        token_type_ids = None,
        attention_mask = None,
    ) :

        sequence_output, pooled_output = self.roberta(
            input_ids = input_ids,
            position_ids = position_ids,
            token_type_ids = token_type_ids,
            attention_mask=attention_mask,
        )
        lm_loss = None

        prediction_scores, seq_relationship_score = self.lm_head(sequence_output)

        return CausalLMOutputWithCrossAttentions(
            loss=lm_loss,
            logits=prediction_scores,
            past_key_values=None,
            hidden_states=None,
            attentions=None,
            cross_attentions=None,
        )

    def from_pretrained(model_id_or_path: str,max_batch, data_type):
        """from_pretrained"""
        torch.set_grad_enabled(False)
        model_dict = {}
        embedding_dict = {}
        torch_model = RobertaForCausalLM.from_pretrained(model_id_or_path)
        roberta = EETRobertaModel.from_torch(torch_model,max_batch,data_type)
        lm_head = torch_model.lm_head.cuda()
        model =  EETRobertaForCausalLM(roberta, lm_head,torch_model.config)

        return model

class EETRobertaForMaskedLM():
    def __init__(self,roberta,lm_head,config):
        self.config = config
        self.roberta = roberta
        self.lm_head = lm_head

    def __call__(
        self,
        input_ids,
        position_ids = None,
        token_type_ids = None,
        attention_mask = None,
    ) :

        sequence_output, pooled_output = self.roberta(
            input_ids = input_ids,
            position_ids = position_ids,
            token_type_ids = token_type_ids,
            attention_mask=attention_mask,
        )

        prediction_scores = self.lm_head(sequence_output)

        # return prediction_scores
        return MaskedLMOutput(
            loss=None,
            logits=prediction_scores,
            hidden_states=None,
            attentions=None,
        )

    def from_pretrained(model_id_or_path: str,max_batch, data_type):
        """from_pretrained"""
        torch.set_grad_enabled(False)
        model_dict = {}
        embedding_dict = {}
        torch_model = RobertaForMaskedLM.from_pretrained(model_id_or_path)
        roberta = EETRobertaModel.from_torch(torch_model,max_batch,data_type)
        lm_head = torch_model.lm_head.cuda()
        model =  EETRobertaForMaskedLM(roberta, lm_head,torch_model.config)

        return model

class EETRobertaForSequenceClassification():
    def __init__(self,roberta,classifier,config):
        self.config = config
        self.roberta = roberta
        self.classifier = classifier

    def __call__(
        self,
        input_ids,
        position_ids = None,
        token_type_ids = None,
        attention_mask = None,
    ) :
        sequence_output, pooled_output = self.roberta(
            input_ids = input_ids,
            position_ids = position_ids,
            token_type_ids = token_type_ids,
            attention_mask=attention_mask,
        )

        logits = self.classifier(sequence_output)

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
        torch_model = RobertaForSequenceClassification.from_pretrained(model_id_or_path)
        roberta = EETRobertaModel.from_torch(torch_model,max_batch,data_type)
        classifier = torch_model.classifier.cuda()
        model =  EETRobertaForSequenceClassification(roberta, classifier,torch_model.config)

        return model

class EETRobertaForMultipleChoice():
    def __init__(self,roberta,classifier,config):
        self.config = config
        self.roberta = roberta
        self.classifier = classifier

    def __call__(
        self,
        input_ids,
        position_ids = None,
        token_type_ids = None,
        attention_mask = None,
    ) :
        num_choices = input_ids.shape[1]
        sequence_output, pooled_output = self.roberta(
            input_ids = input_ids,
            position_ids = position_ids,
            token_type_ids = token_type_ids,
            attention_mask=attention_mask,
        )
        MultipleChoice_pooled_output = pooled_output

        logits = self.classifier(MultipleChoice_pooled_output)
        reshaped_logits = logits.view(-1, num_choices)

        return MultipleChoiceModelOutput(
            loss=None,
            logits=reshaped_logits,
            hidden_states=None,
            attentions=None,
        )

    def from_pretrained(model_id_or_path: str,max_batch, data_type):
        """from_pretrained"""
        torch.set_grad_enabled(False)
        model_dict = {}
        embedding_dict = {}
        torch_model = RobertaForMultipleChoice.from_pretrained(model_id_or_path)
        roberta = EETRobertaModel.from_torch(torch_model,max_batch,data_type)
        classifier = torch_model.classifier.cuda()
        model =  EETRobertaForMultipleChoice(roberta, classifier,torch_model.config)

        return model

class EETRobertaForTokenClassification():
    def __init__(self,roberta,classifier,config):
        self.config = config
        self.roberta = roberta
        self.classifier = classifier

    def __call__(
        self,
        input_ids,
        position_ids = None,
        token_type_ids = None,
        attention_mask = None,
    ) :
        sequence_output, pooled_output = self.roberta(
            input_ids = input_ids,
            position_ids = position_ids,
            token_type_ids = token_type_ids,
            attention_mask=attention_mask,
        )
        MultipleChoice_sequence_output = sequence_output

        logits = self.classifier(MultipleChoice_sequence_output)

        return TokenClassifierOutput(
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
        torch_model = RobertaForTokenClassification.from_pretrained(model_id_or_path)
        roberta = EETRobertaModel.from_torch(torch_model,max_batch,data_type)
        classifier = torch_model.classifier.cuda()
        model =  EETRobertaForTokenClassification(roberta, classifier,torch_model.config)
        return model

class EETRobertaForQuestionAnswering():
    def __init__(self,roberta,qa_outputs,config):
        self.config = config
        self.roberta = roberta
        self.qa_outputs = qa_outputs

    def __call__(
        self,
        input_ids,
        position_ids = None,
        token_type_ids = None,
        attention_mask = None,
    ) :
        sequence_output, pooled_output = self.roberta(
            input_ids = input_ids,
            position_ids = position_ids,
            token_type_ids = token_type_ids,
            attention_mask=attention_mask,
        )
        QuestionAnswering_sequence_output = sequence_output

        logits = self.qa_outputs(QuestionAnswering_sequence_output)
        start_logits, end_logits = logits.split(1, dim=-1)
        start_logits = start_logits.squeeze(-1).contiguous()
        end_logits = end_logits.squeeze(-1).contiguous()

        return QuestionAnsweringModelOutput(
            loss=None,
            start_logits=start_logits,
            end_logits=end_logits,
            hidden_states=None,
            attentions=None,
        )

    def from_pretrained(model_id_or_path: str,max_batch, data_type):
        """from_pretrained"""
        torch.set_grad_enabled(False)
        model_dict = {}
        embedding_dict = {}
        torch_model = RobertaForQuestionAnswering.from_pretrained(model_id_or_path)
        roberta = EETRobertaModel.from_torch(torch_model,max_batch,data_type)
        qa_outputs = torch_model.qa_outputs.cuda()
        model =  EETRobertaForQuestionAnswering(roberta, qa_outputs,torch_model.config)
        return model