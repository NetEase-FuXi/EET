#
# Created by djz on 2021/01/21.
#
"""EET transformers bert model. """

import math
import time
import torch
import torch.nn as nn
import numpy as np
from torch import Tensor
from typing import Any, Dict, List, Optional, Tuple
from transformers import (
    MegatronBertModel,
    MegatronBertForPreTraining,
    MegatronBertForMaskedLM,
    MegatronBertForNextSentencePrediction,
    MegatronBertForSequenceClassification,
    MegatronBertForMultipleChoice,
    MegatronBertForTokenClassification,
    MegatronBertForQuestionAnswering,
)
from transformers.file_utils import ModelOutput
from transformers.modeling_outputs import (
    CausalLMOutputWithCrossAttentions,
    MaskedLMOutput,
    MultipleChoiceModelOutput,
    NextSentencePredictorOutput,
    QuestionAnsweringModelOutput,
    SequenceClassifierOutput,
    TokenClassifierOutput,
)
from eet.transformers.encoder_decoder import *
from eet.utils.mapping import convert_name

from EET import MetaDesc as meta_desc
from EET import FeedForwardNetwork as eet_ffn
from EET import MultiHeadAttention as eet_attention
from EET import Embedding as eet_embedding
from EET import LayerNorm as eet_layernorm

__all__ = ['EETMegatronBertEmbedding', 'EETMegatronBertModel', 'EETMegatronBertForPreTraining',  'EETMegatronBertForMaskedLM',
           'EETMegatronBertForNextSentencePrediction', 'EETMegatronBertForSequenceClassification', 'EETMegatronBertForMultipleChoice',
           'EETMegatronBertForTokenClassification', 'EETMegatronBertForQuestionAnswering']


class MegatronBertForPreTrainingOutput(ModelOutput):
    prediction_logits: torch.FloatTensor = None
    seq_relationship_logits: torch.FloatTensor = None


class EETMegatronBertEmbedding():
    def __init__(self, config, embedding_dict, data_type=torch.float32, name='emb_cache'):
        self.if_layernorm = False
        self.embedding_weights = embedding_dict['embeddings.word_embeddings.weight'].cuda().type(data_type)
        self.position_weights = embedding_dict['embeddings.position_embeddings.weight'].cuda().type(data_type)
        self.token_type_weights = embedding_dict['embeddings.token_type_embeddings.weight'].cuda().type(data_type)
        self.Layernorm_weights = torch.empty(0)
        self.Layernorm_bias = torch.empty(0)
        self.embedding = eet_embedding(config, self.embedding_weights, self.position_weights, self.token_type_weights, self.Layernorm_weights, self.Layernorm_bias, name)

    def __call__(self,
                 input_ids,
                 position_ids,
                 token_type_ids):
        return self.embedding.forward_transformers(input_ids, position_ids, token_type_ids, self.if_layernorm)

    @staticmethod
    def from_torch(config, embedding_dict, data_type=torch.float32, name='emb_cache'):
        embedding = EETMegatronBertEmbedding(config, embedding_dict, data_type=data_type, name=name)
        return embedding


class EETMegatronBertModel():
    def __init__(self, config, embedding,layer_norm, encoder, pooler=None):
        self.embedding = embedding
        self.layernorm = layer_norm
        self.encoder = encoder
        if pooler is not None:
            pooler = pooler.cuda()
        self.pooler = pooler
        self.pre_padding_len = torch.empty(0).long()
        self.position_ids = torch.arange(0, config.max_position_embeddings).reshape(1, config.max_position_embeddings).cuda()

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

        position_ids = self.position_ids[:, :input_shape[1]]

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
                                       normalize_before=True)
        
        layernorm_output = self.layernorm(sequence_output)

        pooled_output = self.pooler(layernorm_output) if self.pooler is not None else None
        return layernorm_output, pooled_output

    @staticmethod
    def from_pretrained(model_id_or_path: str, max_batch, data_type, device_id=0):
        """from_pretrained."""
        torch.set_grad_enabled(False)
        model_dict = {}
        embedding_dict = {}
        torch_model = MegatronBertModel.from_pretrained(model_id_or_path)
        cfg = torch_model.config
        model_name = 'megatronbert'  # cfg.model_type

        for k, v in torch_model.state_dict().items():
            if 'embeddings.' in k:
                embedding_dict[k] = v
            if 'layer.' in k:
                # Structure mapping
                k = convert_name(k, model_name)
                k = k[k.find('layer.'):]
                model_dict[k] = v

        # Group by layer id in model_dict's keys
        from itertools import groupby
        layer_model_dict = {k: dict(v) for k, v in groupby(list(model_dict.items()),
                                                           lambda item: item[0][:(item[0].index('.', item[0].index('.')+1))])}

        device = "cpu" if device_id < 0 else f"cuda:{device_id}"
        activation_fn = cfg.hidden_act
        batch_size = max_batch
        config = meta_desc(dtype=data_type,
                           batch_size=batch_size,
                           head_num=cfg.num_attention_heads,
                           hidden_units=cfg.hidden_size,
                           layer_num=cfg.num_hidden_layers,
                           max_seq_len=cfg.max_position_embeddings,
                           activation_fn=activation_fn,
                           cuda_device=device)

        embedding = EETMegatronBertEmbedding.from_torch(config, embedding_dict, data_type)
        # embedding = None
        encoder = EETEncoder.from_torch(config, layer_model_dict, cfg.num_hidden_layers, data_type)
        if data_type == torch.float32:
            torch_model = torch_model.float()
        else:
            torch_model = torch_model.half()
        eet_model = EETMegatronBertModel(cfg, embedding, encoder, torch_model.pooler)
        return eet_model

    def from_torch(torch_model, max_batch, data_type, device_id=0):
        """from torch."""
        torch.set_grad_enabled(False)
        model_dict = {}
        embedding_dict = {}
        ln_dict = {}

        cfg = torch_model.config
        model_name = 'megatronbert'  # cfg.model_type

        for k, v in torch_model.bert.state_dict().items():
            if 'embeddings.' in k:
                embedding_dict[k] = v
            if 'layer.' in k:
                # Structure mapping
                k = convert_name(k, model_name)
                k = k[k.find('layer.'):]
                model_dict[k] = v
            if 'encoder.ln.' in k:
                ln_dict[k] = v

        # group by 'layer.n'
        from itertools import groupby
        layer_model_dict = {k: dict(v) for k, v in groupby(list(model_dict.items()),
                                                           lambda item: item[0][:(item[0].index('.', item[0].index('.')+1))])}

        device = "cpu" if device_id < 0 else f"cuda:{device_id}"
        activation_fn = cfg.hidden_act
        batch_size = max_batch
        config = meta_desc(dtype=data_type,
                           batch_size=batch_size,
                           head_num=cfg.num_attention_heads,
                           hidden_units=cfg.hidden_size,
                           layer_num=cfg.num_hidden_layers,
                           max_seq_len=cfg.max_position_embeddings,
                           activation_fn=activation_fn,
                           cuda_device=device)

        embedding = EETMegatronBertEmbedding.from_torch(config, embedding_dict, data_type)
        layer_norm = EETLayerNorm.from_torch(config, ln_dict['encoder.ln.weight'], ln_dict['encoder.ln.bias'], data_type)

        # embedding = None
        encoder = EETEncoder.from_torch(config, layer_model_dict, cfg.num_hidden_layers, data_type)
        if data_type == torch.float32:
            torch_model = torch_model.float()
        else:
            torch_model = torch_model.half()
        eet_model = EETMegatronBertModel(cfg, embedding,layer_norm, encoder, torch_model.bert.pooler)
        return eet_model


class EETMegatronBertForPreTraining():
    def __init__(self, bert, cls, config):
        self.config = config
        self.bert = bert
        self.cls = cls

    def __call__(
        self,
        input_ids,
        position_ids=None,
        token_type_ids=None,
        attention_mask=None,
    ):

        sequence_output, pooled_output = self.bert(
            input_ids=input_ids,
            position_ids=position_ids,
            token_type_ids=token_type_ids,
            attention_mask=attention_mask,
        )

        prediction_scores, seq_relationship_score = self.cls(sequence_output, pooled_output)

        return MegatronBertForPreTrainingOutput(
            prediction_logits=prediction_scores,
            seq_relationship_logits=seq_relationship_score,
        )

    def from_pretrained(model_id_or_path: str, max_batch, data_type):
        """from_pretrained"""
        torch.set_grad_enabled(False)
        model_dict = {}
        embedding_dict = {}
        torch_model = MegatronBertForPreTraining.from_pretrained(model_id_or_path)
        bert = EETMegatronBertModel.from_torch(torch_model, max_batch, data_type)
        cls = torch_model.cls.cuda()
        model = EETMegatronBertForPreTraining(bert, cls, torch_model.config)

        return model



class EETMegatronBertForMaskedLM():
    def __init__(self, bert, cls, config):
        self.config = config
        self.bert = bert
        self.cls = cls

    def __call__(
        self,
        input_ids,
        position_ids=None,
        token_type_ids=None,
        attention_mask=None,
    ):

        sequence_output, pooled_output = self.bert(
            input_ids=input_ids,
            position_ids=position_ids,
            token_type_ids=token_type_ids,
            attention_mask=attention_mask,
        )

        prediction_scores = self.cls(sequence_output)

        # return prediction_scores
        return MaskedLMOutput(
            loss=None,
            logits=prediction_scores,
            hidden_states=None,
            attentions=None,
        )

    def from_pretrained(model_id_or_path: str, max_batch, data_type):
        """from_pretrained"""
        torch.set_grad_enabled(False)
        model_dict = {}
        embedding_dict = {}
        torch_model = MegatronBertForMaskedLM.from_pretrained(model_id_or_path)
        bert = EETMegatronBertModel.from_torch(torch_model, max_batch, data_type)
        cls = torch_model.cls.cuda()
        model = EETMegatronBertForMaskedLM(bert, cls, torch_model.config)

        return model


class EETMegatronBertForNextSentencePrediction():
    def __init__(self, bert, cls, config):
        self.config = config
        self.bert = bert
        self.cls = cls

    def __call__(
        self,
        input_ids,
        position_ids=None,
        token_type_ids=None,
        attention_mask=None,
    ):

        sequence_output, pooled_output = self.bert(
            input_ids=input_ids,
            position_ids=position_ids,
            token_type_ids=token_type_ids,
            attention_mask=attention_mask,
        )
        NextSentence_pooled_output = pooled_output

        seq_relationship_scores = self.cls(NextSentence_pooled_output)

        return NextSentencePredictorOutput(
            loss=None,
            logits=seq_relationship_scores,
            hidden_states=None,
            attentions=None,
        )

    def from_pretrained(model_id_or_path: str, max_batch, data_type):
        """from_pretrained"""
        torch.set_grad_enabled(False)
        model_dict = {}
        embedding_dict = {}
        torch_model = MegatronBertForNextSentencePrediction.from_pretrained(model_id_or_path)
        bert = EETMegatronBertModel.from_torch(torch_model, max_batch, data_type)
        cls = torch_model.cls.cuda()
        model = EETMegatronBertForNextSentencePrediction(bert, cls, torch_model.config)

        return model


class EETMegatronBertForSequenceClassification():
    def __init__(self, bert, classifier, config):
        self.config = config
        self.bert = bert
        self.classifier = classifier

    def __call__(
        self,
        input_ids,
        position_ids=None,
        token_type_ids=None,
        attention_mask=None,
    ):
        sequence_output, pooled_output = self.bert(
            input_ids=input_ids,
            position_ids=position_ids,
            token_type_ids=token_type_ids,
            attention_mask=attention_mask,
        )
        SequenceClassification_pooled_output = pooled_output

        logits = self.classifier(SequenceClassification_pooled_output)

        return SequenceClassifierOutput(
            loss=None,
            logits=logits,
            hidden_states=None,
            attentions=None,
        )

    def from_pretrained(model_id_or_path: str, max_batch, data_type):
        """from_pretrained"""
        torch.set_grad_enabled(False)
        model_dict = {}
        embedding_dict = {}
        torch_model = MegatronBertForSequenceClassification.from_pretrained(model_id_or_path)
        bert = EETMegatronBertModel.from_torch(torch_model, max_batch, data_type)
        classifier = torch_model.classifier.cuda()
        model = EETMegatronBertForSequenceClassification(bert, classifier, torch_model.config)

        return model


class EETMegatronBertForMultipleChoice():
    def __init__(self, bert, classifier, config):
        self.config = config
        self.bert = bert
        self.classifier = classifier

    def __call__(
        self,
        input_ids,
        position_ids=None,
        token_type_ids=None,
        attention_mask=None,
    ):
        num_choices = input_ids.shape[1]
        sequence_output, pooled_output = self.bert(
            input_ids=input_ids,
            position_ids=position_ids,
            token_type_ids=token_type_ids,
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

    def from_pretrained(model_id_or_path: str, max_batch, data_type):
        """from_pretrained"""
        torch.set_grad_enabled(False)
        model_dict = {}
        embedding_dict = {}
        torch_model = MegatronBertForMultipleChoice.from_pretrained(model_id_or_path)
        bert = EETMegatronBertModel.from_torch(torch_model, max_batch, data_type)
        classifier = torch_model.classifier.cuda()
        model = EETMegatronBertForMultipleChoice(bert, classifier, torch_model.config)

        return model


class EETMegatronBertForTokenClassification():
    def __init__(self, bert, classifier, config):
        self.config = config
        self.bert = bert
        self.classifier = classifier

    def __call__(
        self,
        input_ids,
        position_ids=None,
        token_type_ids=None,
        attention_mask=None,
    ):
        sequence_output, pooled_output = self.bert(
            input_ids=input_ids,
            position_ids=position_ids,
            token_type_ids=token_type_ids,
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

    def from_pretrained(model_id_or_path: str, max_batch, data_type):
        """from_pretrained"""
        torch.set_grad_enabled(False)
        model_dict = {}
        embedding_dict = {}
        torch_model = MegatronBertForTokenClassification.from_pretrained(model_id_or_path)
        bert = EETMegatronBertModel.from_torch(torch_model, max_batch, data_type)
        classifier = torch_model.classifier.cuda()
        model = EETMegatronBertForTokenClassification(bert, classifier, torch_model.config)
        return model


class EETMegatronBertForQuestionAnswering():
    def __init__(self, bert, qa_outputs, config):
        self.config = config
        self.bert = bert
        self.qa_outputs = qa_outputs

    def __call__(
        self,
        input_ids,
        position_ids=None,
        token_type_ids=None,
        attention_mask=None,
    ):
        sequence_output, pooled_output = self.bert(
            input_ids=input_ids,
            position_ids=position_ids,
            token_type_ids=token_type_ids,
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

    def from_pretrained(model_id_or_path: str, max_batch, data_type):
        """from_pretrained"""
        torch.set_grad_enabled(False)
        model_dict = {}
        embedding_dict = {}
        torch_model = MegatronBertForQuestionAnswering.from_pretrained(model_id_or_path)
        bert = EETMegatronBertModel.from_torch(torch_model, max_batch, data_type)
        qa_outputs = torch_model.qa_outputs.cuda()
        model = EETMegatronBertForQuestionAnswering(bert, qa_outputs, torch_model.config)
        return model
