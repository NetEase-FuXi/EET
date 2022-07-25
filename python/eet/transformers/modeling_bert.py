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
from transformers import  (
    BertModel,
    BertForPreTraining,
    BertLMHeadModel,
    BertForMaskedLM,
    BertForNextSentencePrediction,
    BertForSequenceClassification,
    BertForMultipleChoice,
    BertForTokenClassification,
    BertForQuestionAnswering,
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

__all__ = ['EETBertEmbedding', 'EETBertModel', 'EETBertForPreTraining', 'EETBertLMHeadModel', 'EETBertForMaskedLM',
             'EETBertForNextSentencePrediction', 'EETBertForSequenceClassification', 'EETBertForMultipleChoice', 
             'EETBertForTokenClassification', 'EETBertForQuestionAnswering']
class BertForPreTrainingOutput(ModelOutput):
    prediction_logits: torch.FloatTensor = None
    seq_relationship_logits: torch.FloatTensor = None

class EETBertEmbedding():
    def __init__(self, config, embedding_dict, data_type=torch.float32, name='emb_cache'):
        self.if_layernorm = True
        self.embedding_weights = embedding_dict['embeddings.word_embeddings.weight'].cuda().type(data_type)
        self.position_weights = embedding_dict['embeddings.position_embeddings.weight'].cuda().type(data_type)
        self.token_type_weights = embedding_dict['embeddings.token_type_embeddings.weight'].cuda().type(data_type)
        self.Layernorm_weights = embedding_dict['embeddings.LayerNorm.weight'].cuda().type(data_type)
        self.Layernorm_bias = embedding_dict['embeddings.LayerNorm.bias'].cuda().type(data_type)
        self.embedding = eet_embedding(config,self.embedding_weights,self.position_weights,self.token_type_weights,self.Layernorm_weights,self.Layernorm_bias, name)
    
    def __call__(self,
                input_ids,
                position_ids,
                token_type_ids):
        return self.embedding.forward_transformers(input_ids,position_ids,token_type_ids,self.if_layernorm)
    
    @staticmethod
    def from_torch(config, embedding_dict, data_type=torch.float32, name='emb_cache'):
        embedding = EETBertEmbedding(config, embedding_dict, data_type=data_type, name=name)
        return embedding


class EETBertModel():
    def __init__(self,config,embedding,encoder, pooler=None):
        self.embedding = embedding
        self.encoder = encoder
        if pooler is not None:
            pooler = pooler.cuda()
        self.pooler = pooler
        self.pre_padding_len = torch.empty(0).long()
        self.position_ids = torch.arange(0,config.max_position_embeddings).reshape(1,config.max_position_embeddings).cuda()
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
        
        sequence_output = self.encoder(embedding_out,
                    pre_padding_len = pre_padding_len,
                    normalize_before = False)
        pooled_output = self.pooler(sequence_output) if self.pooler is not None else None
        return sequence_output,pooled_output
    
    @staticmethod
    def from_pretrained(model_id_or_path: str,max_batch, data_type, device_id=0):
        """from_pretrained."""
        torch.set_grad_enabled(False)
        model_dict = {}
        embedding_dict = {}
        torch_model = BertModel.from_pretrained(model_id_or_path)
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

        # Group by layer id in model_dict's keys
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
        eet_model = EETBertModel(cfg, embedding, encoder,torch_model.pooler)
        return eet_model

    def from_torch(torch_model,max_batch, data_type, device_id=0):
        """from torch."""
        torch.set_grad_enabled(False)
        model_dict = {}
        embedding_dict = {}
        cfg = torch_model.config
        model_name = cfg.model_type
        
        for k, v in torch_model.bert.state_dict().items():
            if 'embeddings.' in k:
                embedding_dict[k] = v
            if 'layer.' in k:
                # Structure mapping
                k = convert_name(k, model_name)
                k = k[k.find('layer.'):]
                model_dict[k] = v

        # group by 'layer.n'
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
        eet_model = EETBertModel(cfg, embedding, encoder,torch_model.bert.pooler)
        return eet_model


class EETBertForPreTraining():
    def __init__(self,bert,cls,config):
        self.config = config
        self.bert = bert
        self.cls = cls

    def __call__(
        self,
        input_ids,
        position_ids = None,
        token_type_ids = None,
        attention_mask = None,
    ) :

        sequence_output, pooled_output = self.bert(
            input_ids = input_ids,
            position_ids = position_ids,
            token_type_ids = token_type_ids,
            attention_mask=attention_mask,
        )

        prediction_scores, seq_relationship_score = self.cls(sequence_output, pooled_output)

        return BertForPreTrainingOutput(
            prediction_logits=prediction_scores,
            seq_relationship_logits=seq_relationship_score,
        )

    def from_pretrained(model_id_or_path: str,max_batch, data_type):
        """from_pretrained"""
        torch.set_grad_enabled(False)
        model_dict = {}
        embedding_dict = {}
        torch_model = BertForPreTraining.from_pretrained(model_id_or_path)
        bert = EETBertModel.from_torch(torch_model,max_batch,data_type)
        cls = torch_model.cls.cuda()
        model =  EETBertForPreTraining(bert, cls,torch_model.config)

        return model

class EETBertLMHeadModel():
    def __init__(self,bert,cls,config):
        self.config = config
        self.bert = bert
        self.cls = cls

    def __call__(
        self,
        input_ids,
        position_ids = None,
        token_type_ids = None,
        attention_mask = None,
    ) :

        sequence_output, pooled_output = self.bert(
            input_ids = input_ids,
            position_ids = position_ids,
            token_type_ids = token_type_ids,
            attention_mask=attention_mask,
        )

        prediction_scores = self.cls(sequence_output)

        return CausalLMOutputWithCrossAttentions(
            loss=None,
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
        torch_model = BertLMHeadModel.from_pretrained(model_id_or_path)
        bert = EETBertModel.from_torch(torch_model,max_batch,data_type)
        cls = torch_model.cls.cuda()
        model =  EETBertLMHeadModel(bert, cls,torch_model.config)

        return model

class EETBertForMaskedLM():
    def __init__(self,bert,cls,config):
        self.config = config
        self.bert = bert
        self.cls = cls

    def __call__(
        self,
        input_ids,
        position_ids = None,
        token_type_ids = None,
        attention_mask = None,
    ) :

        sequence_output, pooled_output = self.bert(
            input_ids = input_ids,
            position_ids = position_ids,
            token_type_ids = token_type_ids,
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

    def from_pretrained(model_id_or_path: str,max_batch, data_type):
        """from_pretrained"""
        torch.set_grad_enabled(False)
        model_dict = {}
        embedding_dict = {}
        torch_model = BertForMaskedLM.from_pretrained(model_id_or_path)
        bert = EETBertModel.from_torch(torch_model,max_batch,data_type)
        cls = torch_model.cls.cuda()
        model =  EETBertForMaskedLM(bert, cls,torch_model.config)

        return model

class EETBertForNextSentencePrediction():
    def __init__(self,bert,cls,config):
        self.config = config
        self.bert = bert
        self.cls = cls

    def __call__(
        self,
        input_ids,
        position_ids = None,
        token_type_ids = None,
        attention_mask = None,
    ) :

        sequence_output, pooled_output = self.bert(
            input_ids = input_ids,
            position_ids = position_ids,
            token_type_ids = token_type_ids,
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

    def from_pretrained(model_id_or_path: str,max_batch, data_type):
        """from_pretrained"""
        torch.set_grad_enabled(False)
        model_dict = {}
        embedding_dict = {}
        torch_model = BertForNextSentencePrediction.from_pretrained(model_id_or_path)
        bert = EETBertModel.from_torch(torch_model,max_batch,data_type)
        cls = torch_model.cls.cuda()
        model =  EETBertForNextSentencePrediction(bert, cls,torch_model.config)

        return model

class EETBertForSequenceClassification():
    def __init__(self,bert,classifier,config):
        self.config = config
        self.bert = bert
        self.classifier = classifier

    def __call__(
        self,
        input_ids,
        position_ids = None,
        token_type_ids = None,
        attention_mask = None,
    ) :
        sequence_output, pooled_output = self.bert(
            input_ids = input_ids,
            position_ids = position_ids,
            token_type_ids = token_type_ids,
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

    def from_pretrained(model_id_or_path: str,max_batch, data_type):
        """from_pretrained"""
        torch.set_grad_enabled(False)
        model_dict = {}
        embedding_dict = {}
        torch_model = BertForSequenceClassification.from_pretrained(model_id_or_path)
        bert = EETBertModel.from_torch(torch_model,max_batch,data_type)
        classifier = torch_model.classifier.cuda()
        model =  EETBertForSequenceClassification(bert, classifier,torch_model.config)

        return model

class EETBertForMultipleChoice():
    def __init__(self,bert,classifier,config):
        self.config = config
        self.bert = bert
        self.classifier = classifier

    def __call__(
        self,
        input_ids,
        position_ids = None,
        token_type_ids = None,
        attention_mask = None,
    ) :
        num_choices = input_ids.shape[1]
        sequence_output, pooled_output = self.bert(
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
        torch_model = BertForMultipleChoice.from_pretrained(model_id_or_path)
        bert = EETBertModel.from_torch(torch_model,max_batch,data_type)
        classifier = torch_model.classifier.cuda()
        model =  EETBertForMultipleChoice(bert, classifier,torch_model.config)

        return model

class EETBertForTokenClassification():
    def __init__(self,bert,classifier,config):
        self.config = config
        self.bert = bert
        self.classifier = classifier

    def __call__(
        self,
        input_ids,
        position_ids = None,
        token_type_ids = None,
        attention_mask = None,
    ) :
        sequence_output, pooled_output = self.bert(
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
        torch_model = BertForTokenClassification.from_pretrained(model_id_or_path)
        bert = EETBertModel.from_torch(torch_model,max_batch,data_type)
        classifier = torch_model.classifier.cuda()
        model =  EETBertForTokenClassification(bert, classifier,torch_model.config)
        return model

class EETBertForQuestionAnswering():
    def __init__(self,bert,qa_outputs,config):
        self.config = config
        self.bert = bert
        self.qa_outputs = qa_outputs

    def __call__(
        self,
        input_ids,
        position_ids = None,
        token_type_ids = None,
        attention_mask = None,
    ) :
        sequence_output, pooled_output = self.bert(
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
        torch_model = BertForQuestionAnswering.from_pretrained(model_id_or_path)
        bert = EETBertModel.from_torch(torch_model,max_batch,data_type)
        qa_outputs = torch_model.qa_outputs.cuda()
        model =  EETBertForQuestionAnswering(bert, qa_outputs,torch_model.config)
        return model