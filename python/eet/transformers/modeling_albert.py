#
# Created by djz.
#
"""EET transformers albert model. """

import math
import time
import torch
import torch.nn as nn
import torch.nn.functional as F
import numpy as np
from torch import Tensor
from typing import Any, Dict, List, Optional, Tuple
from dataclasses import dataclass
from transformers import  (
    AlbertModel,
    AlbertForPreTraining,
    AlbertForMaskedLM,
    AlbertForSequenceClassification,
    AlbertForTokenClassification,
    AlbertForQuestionAnswering,
    AlbertForMultipleChoice,
)
from transformers.file_utils import ModelOutput
from transformers.modeling_outputs import (
    MaskedLMOutput,
    MultipleChoiceModelOutput,
    QuestionAnsweringModelOutput,
    SequenceClassifierOutput,
    TokenClassifierOutput,
)
from eet.transformers.encoder_decoder import *
from eet.transformers.modeling_bert import EETBertEmbedding
from eet.utils.mapping import convert_name

from EET import MetaDesc as meta_desc
from EET import FeedForwardNetwork as eet_ffn
from EET import MultiHeadAttention as eet_attention
from EET import Embedding as eet_embedding


__all__ = ['EETAlbertEncoder', 'EETAlbertModel', 'EETAlbertForPreTraining', 'EETAlbertForMaskedLM', 'EETAlbertForSequenceClassification', 
            'EETAlbertForTokenClassification', 'EETAlbertForQuestionAnswering', 'EETAlbertForMultipleChoice']

@dataclass
class AlbertForPreTrainingOutput(ModelOutput):
    loss: Optional[torch.FloatTensor] = None
    prediction_logits: torch.FloatTensor = None
    sop_logits: torch.FloatTensor = None
    hidden_states: Optional[Tuple[torch.FloatTensor]] = None
    attentions: Optional[Tuple[torch.FloatTensor]] = None


class EETAlbertGroup():
    def __init__(self, EncoderLayers, config, cfg):
        self.layers = EncoderLayers
        self.config = config
        self.cfg = cfg

    def __call__(
        self,
        x,
        pre_padding_len = None,
        normalize_before = False,
    ):
        for idx, layer in enumerate(self.layers):
            x = layer(
                x,
                pre_padding_len=pre_padding_len,
                normalize_before=False
            )
        return x
    
    @staticmethod
    def from_torch(layer_model_dict, config, cfg, data_type=torch.float32):
        """from torch."""
        EncoderLayers = []
        for i in range(cfg.inner_group_num):
            EncoderLayers.extend(
                [
                    EETEncoderLayer.from_torch(config, layer_model_dict['layer.' + str(i)], i, data_type=data_type)
                ]
            )
        eet_encoder = EETAlbertGroup(EncoderLayers, config, cfg)
        return eet_encoder


class EETAlbertEncoder():
    def __init__(self, EncoderGroups, hidden_mapping_in, config, cfg):
        self.hidden_mapping_in = hidden_mapping_in
        self.groups = EncoderGroups
        self.config = config
        self.cfg = cfg
        self.layers_per_group = int(self.cfg.num_hidden_layers / self.cfg.num_hidden_groups)

    def __call__(
        self,
        x,
        pre_padding_len = None,
        normalize_before = False,
    ):
        x = self.hidden_mapping_in(x)
        for i in range(self.cfg.num_hidden_layers):
            group_idx = int(i / self.layers_per_group)
            x = self.groups[group_idx](
                x,
                pre_padding_len=pre_padding_len,
                normalize_before=False
            )
        return x
    
    @staticmethod
    def from_torch(layer_model_dict, config, cfg, data_type=torch.float32):
        """from torch."""
        hidden_mapping_in = FCLayer(layer_model_dict['hidden_mapping_in.weight'],
                                    layer_model_dict['hidden_mapping_in.bias'], data_type=data_type)

        EncoderGroups = []
        for i in range(cfg.num_hidden_groups):
            EncoderGroups.extend(
                [
                    EETAlbertGroup.from_torch(layer_model_dict['group.' + str(i)], config, cfg, data_type=data_type)
                ]
            )
        eet_encoder = EETAlbertEncoder(EncoderGroups, hidden_mapping_in, config, cfg)
        return eet_encoder


class EETAlbertModel():
    def __init__(self, config, embedding, encoder, pooler_activation, pooler):
        self.embedding = embedding
        self.encoder = encoder
        self.pre_padding_len = torch.empty(0).long()
        self.position_ids = torch.arange(0,config.max_position_embeddings).reshape(1,config.max_position_embeddings).cuda()
        if pooler is not None:
            self.pooler = pooler.cuda()
            self.pooler_activation = pooler_activation.cuda()
        else:
            self.pooler = None
            self.pooler_activation = None
    
    def __call__(
        self,
        input_ids,
        position_ids = None,
        token_type_ids = None,
        attention_mask = None,
    ):
        '''
        attention_mask:attention_padding_mask(:obj:`torch.FloatTensor` of shape :obj:`(batch_size, sequence_length)`, `optional`, defaults to :obj:`None`):
            Mask to avoid performing attention on the padding token indices of the encoder input.)
            Mask values selected in ``[0, 1]``:
            ``1`` for tokens that are NOT MASKED, ``0`` for MASKED tokens.
        '''
        input_shape = input_ids.size()
        position_ids = self.position_ids[:, :input_shape[1]]

        if token_type_ids is None:
            token_type_ids = torch.zeros(input_shape, dtype=torch.long, device=input_ids.device)
        
        if attention_mask is None:
            pre_padding_len = self.pre_padding_len
        else:
            # transformers 0 - padding;1 - nopadding
            pre_padding_len =  torch.sum(1 - attention_mask,1).long().cuda()
            
        embedding_out = self.embedding(input_ids,position_ids,token_type_ids)

        sequence_output = self.encoder(
            embedding_out,
            pre_padding_len=pre_padding_len,
            normalize_before=False
        )

        pooled_output = self.pooler_activation(self.pooler(sequence_output[:, 0])) if self.pooler is not None else None

        return sequence_output, pooled_output

    @staticmethod
    def create_state_dict(torch_model, model_name):
        model_dict = {}
        embedding_dict = {}
        layer_model_dict = {}
        other_dict = {}
        for k, v in torch_model.state_dict().items():
            k = convert_name(k, model_name, verbose=False)
            if 'embeddings.' in k:
                k = k[k.find('embeddings.'):]
                embedding_dict[k] = v
            if 'group.' in k:
                k = k[k.find('group.'):]
                layer_model_dict[k] = v
            if 'hidden_mapping_in' in k:
                k = k[k.find('hidden_mapping_in'):]
                model_dict[k] = v

        # Group by group id first, then group by layer id, e.g. model_dict['group.0']['layer.0'] is a set of param key-value pairs.
        from itertools import groupby
        for k1, v1 in groupby(list(layer_model_dict.items()), lambda item: item[0][:item[0].index('.', item[0].index('.')+1)]):
            temp = []
            for k2, v2 in v1:
                k2 = k2[k2.index('layer.'):]
                temp.append((k2, v2))
            model_dict[k1] = {k: dict(v) for k, v in groupby(temp, lambda item: item[0][:item[0].index('.', item[0].index('.')+1)])}
        model_dict['embeddings'] = embedding_dict
        return model_dict
    
    @staticmethod
    def from_pretrained(model_id_or_path: str,max_batch, data_type, device_id=0):
        """from_pretrained."""    
        torch.set_grad_enabled(False)
        torch_model = AlbertModel.from_pretrained(model_id_or_path)
        cfg = torch_model.config
        model_name = cfg.model_type

        # create eet model state dict
        model_dict = EETAlbertModel.create_state_dict(torch_model, model_name)

        device = "cpu" if device_id < 0 else f"cuda:{device_id}"
        activation_fn = cfg.hidden_act
        batch_size = max_batch
        config = meta_desc(batch_size, cfg.num_attention_heads, cfg.hidden_size, cfg.num_hidden_layers, cfg.max_position_embeddings, cfg.max_position_embeddings, data_type, device, False, activation_fn)
        config_emb = meta_desc(batch_size, cfg.num_attention_heads, cfg.embedding_size, cfg.num_hidden_layers, cfg.max_position_embeddings, cfg.max_position_embeddings, data_type, device, False, activation_fn)

        embedding = EETBertEmbedding.from_torch(config_emb, model_dict['embeddings'], data_type)
        encoder = EETAlbertEncoder.from_torch(model_dict, config, cfg, data_type)
        if data_type == torch.float32:
            torch_model = torch_model.float()
        else:
            torch_model = torch_model.half()
        eet_model =  EETAlbertModel(cfg, embedding, encoder, torch_model.pooler_activation, torch_model.pooler)
        return eet_model

    def from_torch(torch_model, max_batch, data_type, device_id=0):
        """from torch."""
        torch.set_grad_enabled(False)
        cfg = torch_model.config
        model_name = cfg.model_type

        # create eet model state dict
        model_dict = EETAlbertModel.create_state_dict(torch_model, model_name)

        device = "cpu" if device_id < 0 else f"cuda:{device_id}"
        activation_fn = cfg.hidden_act
        batch_size = max_batch
        config = meta_desc(batch_size, cfg.num_attention_heads, cfg.hidden_size, cfg.num_hidden_layers, cfg.max_position_embeddings, cfg.max_position_embeddings, data_type, device, False, activation_fn)
        config_emb = meta_desc(batch_size, cfg.num_attention_heads, cfg.embedding_size, cfg.num_hidden_layers, cfg.max_position_embeddings, cfg.max_position_embeddings, data_type, device, False, activation_fn)

        embedding = EETBertEmbedding.from_torch(config_emb, model_dict['embeddings'], data_type)
        encoder = EETAlbertEncoder.from_torch(model_dict, config, cfg, data_type)
        if data_type == torch.float32:
            torch_model = torch_model.float()
        else:
            torch_model = torch_model.half()
        eet_model = EETAlbertModel(cfg, embedding, encoder, torch_model.albert.pooler_activation, torch_model.albert.pooler)
        return eet_model

class EETAlbertForPreTraining():
    def __init__(self,albert,predictions,sop_classifier,config):
        self.config = config
        self.albert = albert
        self.predictions = predictions
        self.sop_classifier = sop_classifier

    def __call__(
        self,
        input_ids,
        position_ids = None,
        token_type_ids = None,
        attention_mask = None,
    ) :

        sequence_output, pooled_output = self.albert(
            input_ids = input_ids,
            position_ids = position_ids,
            token_type_ids = token_type_ids,
            attention_mask=attention_mask,
        )

        prediction_scores = self.predictions(sequence_output)
        sop_scores = self.sop_classifier(pooled_output)

        return AlbertForPreTrainingOutput(
            loss=None,
            prediction_logits=prediction_scores,
            sop_logits=sop_scores,
            hidden_states=None,
            attentions=None,
        )


    def from_pretrained(model_id_or_path: str,max_batch, data_type):
        """from_pretrained"""
        torch.set_grad_enabled(False)
        model_dict = {}
        embedding_dict = {}
        torch_model = AlbertForPreTraining.from_pretrained(model_id_or_path)
        albert = EETAlbertModel.from_torch(torch_model,max_batch,data_type)
        predictions = torch_model.predictions.cuda()
        sop_classifier = torch_model.sop_classifier.cuda()

        model =  EETAlbertForPreTraining(albert, predictions,sop_classifier,torch_model.config)

        return model

class EETAlbertForMaskedLM():
    def __init__(self,albert,predictions,config):
        self.config = config
        self.albert = albert
        self.predictions = predictions

    def __call__(
        self,
        input_ids,
        position_ids = None,
        token_type_ids = None,
        attention_mask = None,
    ) :

        sequence_output, pooled_output = self.albert(
            input_ids = input_ids,
            position_ids = position_ids,
            token_type_ids = token_type_ids,
            attention_mask=attention_mask,
        )
        prediction_scores = self.predictions(sequence_output)

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
        torch_model = AlbertForMaskedLM.from_pretrained(model_id_or_path)
        albert = EETAlbertModel.from_torch(torch_model,max_batch,data_type)
        predictions = torch_model.predictions.cuda()
        model =  EETAlbertForMaskedLM(albert, predictions,torch_model.config)

        return model

class EETAlbertForSequenceClassification():
    def __init__(self,albert,classifier,config):
        self.config = config
        self.albert = albert
        self.classifier = classifier

    def __call__(
        self,
        input_ids,
        position_ids = None,
        token_type_ids = None,
        attention_mask = None,
    ) :

        sequence_output, pooled_output = self.albert(
            input_ids = input_ids,
            position_ids = position_ids,
            token_type_ids = token_type_ids,
            attention_mask=attention_mask,
        )
        logits = self.classifier(pooled_output)

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
        torch_model = AlbertForSequenceClassification.from_pretrained(model_id_or_path)
        albert = EETAlbertModel.from_torch(torch_model,max_batch,data_type)
        classifier = torch_model.classifier.cuda()
        model =  EETAlbertForSequenceClassification(albert, classifier,torch_model.config)

        return model

class EETAlbertForTokenClassification():
    def __init__(self,albert,classifier,config):
        self.config = config
        self.albert = albert
        self.classifier = classifier

    def __call__(
        self,
        input_ids,
        position_ids = None,
        token_type_ids = None,
        attention_mask = None,
    ) :

        sequence_output, pooled_output = self.albert(
            input_ids = input_ids,
            position_ids = position_ids,
            token_type_ids = token_type_ids,
            attention_mask=attention_mask,
        )
        logits = self.classifier(sequence_output)

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
        torch_model = AlbertForTokenClassification.from_pretrained(model_id_or_path)
        albert = EETAlbertModel.from_torch(torch_model,max_batch,data_type)
        classifier = torch_model.classifier.cuda()
        model =  EETAlbertForTokenClassification(albert, classifier,torch_model.config)
        return model

class EETAlbertForQuestionAnswering():
    def __init__(self,albert,qa_outputs,config):
        self.config = config
        self.albert = albert
        self.qa_outputs = qa_outputs

    def __call__(
        self,
        input_ids,
        position_ids = None,
        token_type_ids = None,
        attention_mask = None,
    ) :
        sequence_output, pooled_output = self.albert(
            input_ids = input_ids,
            position_ids = position_ids,
            token_type_ids = token_type_ids,
            attention_mask=attention_mask,
        )
        logits = self.qa_outputs(sequence_output)
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
        torch_model = AlbertForQuestionAnswering.from_pretrained(model_id_or_path)
        albert = EETAlbertModel.from_torch(torch_model,max_batch,data_type)
        qa_outputs = torch_model.qa_outputs.cuda()
        model =  EETAlbertForQuestionAnswering(albert, qa_outputs,torch_model.config)
        return model

class EETAlbertForMultipleChoice():
    def __init__(self,albert,classifier,config):
        self.config = config
        self.albert = albert
        self.classifier = classifier

    def __call__(
        self,
        input_ids,
        position_ids = None,
        token_type_ids = None,
        attention_mask = None,
    ) :
        num_choices = input_ids.shape[1]

        sequence_output, pooled_output = self.albert(
            input_ids = input_ids,
            position_ids = position_ids,
            token_type_ids = token_type_ids,
            attention_mask=attention_mask,
        )
        logits = self.classifier(pooled_output)
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
        torch_model = AlbertForMultipleChoice.from_pretrained(model_id_or_path)
        albert = EETAlbertModel.from_torch(torch_model,max_batch,data_type)
        classifier = torch_model.classifier.cuda()
        model =  EETAlbertForMultipleChoice(albert, classifier,torch_model.config)
        return model