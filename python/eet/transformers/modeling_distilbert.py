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
    DistilBertModel,
    DistilBertForMaskedLM,
    DistilBertForSequenceClassification,
    DistilBertForMultipleChoice,
    DistilBertForTokenClassification,
    DistilBertForQuestionAnswering,
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
from eet.transformers.encoder import *

from EET import MetaDesc as meta_desc
from EET import FeedForwardNetwork as eet_ffn
from EET import MultiHeadAttention as eet_attention
from EET import Embedding as eet_embedding
BEGIN_OF_PARAM = 12

__all__ = ['EETDistilBertEmbedding', 'EETDistilBertModel', 'EETDistilBertForMaskedLM','EETDistilBertForSequenceClassification', 'EETDistilBertForMultipleChoice', 
             'EETDistilBertForTokenClassification', 'EETDistilBertForQuestionAnswering']
class DistilBertForPreTrainingOutput(ModelOutput):
    prediction_logits: torch.FloatTensor = None
    seq_relationship_logits: torch.FloatTensor = None

class EETDistilBertEmbedding():
    def __init__(self,config,embedding_dict,data_type = torch.float32):
        self.if_layernorm = True
        self.embedding_weights = embedding_dict['embeddings.word_embeddings.weight'].cuda().type(data_type)
        self.position_weights = embedding_dict['embeddings.position_embeddings.weight'].cuda().type(data_type)
        self.token_type_weights = self.position_weights
        self.Layernorm_weights = embedding_dict['embeddings.LayerNorm.weight'].cuda().type(data_type)
        self.Layernorm_bias = embedding_dict['embeddings.LayerNorm.bias'].cuda().type(data_type)
        self.embedding = eet_embedding(config,self.embedding_weights,self.position_weights,self.token_type_weights,self.Layernorm_weights,self.Layernorm_bias)
    def __call__(self,
                input_ids,
                position_ids,
                token_type_ids):
        return self.embedding.forward_transformers(input_ids,position_ids,token_type_ids,self.if_layernorm)
    
    @staticmethod
    def from_torch(config,embedding_dict,data_type = torch.float32):
        feedforward = EETDistilBertEmbedding(config,embedding_dict,data_type = data_type)
        return feedforward

class EETDistilBertFeedforward():
    def __init__(self,config,model_dict,layer_id,data_type = torch.float32):
        self.intermediate_weights = torch.t([x[1] for x in model_dict.items() if 'ffn.lin1.weight' in x[0]][0]).contiguous().cuda().type(data_type)
        self.intermediate_bias = [x[1] for x in model_dict.items() if 'ffn.lin1.bias' in x[0]][0].cuda().type(data_type)
        self.output_weights = torch.t([x[1] for x in model_dict.items() if 'ffn.lin2.weight' in x[0]][0]).contiguous().cuda().type(data_type)
        self.output_bias = [x[1] for x in model_dict.items() if 'ffn.lin2.bias' in x[0]][0].cuda().type(data_type)
        self.layernorm_weights = [x[1] for x in model_dict.items() if 'output_layer_norm.weight' in x[0]][0].cuda().type(data_type)
        self.layernorm_bias = [x[1] for x in model_dict.items() if 'output_layer_norm.bias' in x[0]][0].cuda().type(data_type)

        self.ffn = eet_ffn(config,self.intermediate_weights,self.intermediate_bias,self.output_weights,self.output_bias,self.layernorm_weights,self.layernorm_bias)
    def __call__(self,
                input_id,
                pre_layernorm = True,
                add_redusial = True):
        return self.ffn.forward(input_id,pre_layernorm,add_redusial)
    
    @staticmethod
    def from_torch(config,model_dict,layer_id,data_type = torch.float32):
        feedforward = EETDistilBertFeedforward(config,model_dict,layer_id,data_type = data_type)
        return feedforward

class EETDistilBertAttention():
    def __init__(self,config, model_dict,layer_id,data_type = torch.float32):
        q_weights = [x[1] for x in model_dict.items() if 'attention.q_lin.weight' in x[0]][0].contiguous().cuda().type(data_type)
        k_weights = [x[1] for x in model_dict.items() if 'attention.k_lin.weight' in x[0]][0].contiguous().cuda().type(data_type)
        v_weights = [x[1] for x in model_dict.items() if 'attention.v_lin.weight' in x[0]][0].contiguous().cuda().type(data_type)
        self.qkv_weight = torch.cat((q_weights,k_weights,v_weights),0).transpose(0,1).contiguous()
        self.q_bias = [x[1] for x in model_dict.items() if 'attention.q_lin.bias' in x[0]][0].cuda().type(data_type)
        self.k_bias = [x[1] for x in model_dict.items() if 'attention.k_lin.bias' in x[0]][0].cuda().type(data_type)
        self.v_bias = [x[1] for x in model_dict.items() if 'attention.v_lin.bias' in x[0]][0].cuda().type(data_type)
        self.out_weights = torch.t([x[1] for x in model_dict.items() if 'attention.out_lin.weight' in x[0]][0]).contiguous().cuda().type(data_type)
        self.out_bias = [x[1] for x in model_dict.items() if 'attention.out_lin.bias' in x[0]][0].cuda().type(data_type)
        self.layernorm_weights = [x[1] for x in model_dict.items() if 'sa_layer_norm.weight' in x[0]][0].cuda().type(data_type)
        self.layernorm_bias = [x[1] for x in model_dict.items() if 'sa_layer_norm.bias' in x[0]][0].cuda().type(data_type)

        self.attention = eet_attention(config,self.qkv_weight,self.q_bias,self.k_bias,self.v_bias,self.out_weights,self.out_bias,self.layernorm_weights,self.layernorm_bias)

    def __call__(self,
                input_id,
                pre_padding_len,
                pre_layernorm = False,
                add_redusial = True,
                need_sequence_mask=False,):
        return self.attention.forward(input_id,pre_padding_len,pre_layernorm,add_redusial,need_sequence_mask)

    @staticmethod
    def from_torch(config,model_dict,layer_id,data_type = torch.float32):
        attention = EETDistilBertAttention(config,model_dict,layer_id,data_type = data_type)
        return attention

class EETDistilBertEncoderLayer():
    def __init__(self, config, attention,feedforward):
        self.attetion = attention
        self.feedforward = feedforward

    def __call__(self,
                x,
                pre_padding_len = None,
                normalize_before = False):

        ''' gpt2 model struct '''
        ''' layernorm->self_attention-> project->addinputbias->layernorm->ffn->addinputbias'''
        self_attn_out = self.attetion(input_id = x,
                    pre_padding_len = pre_padding_len,
                    pre_layernorm = normalize_before,
                    add_redusial = True)
        out = self.feedforward(self_attn_out,
                    pre_layernorm = normalize_before,
                    add_redusial = True)

        return out

    @staticmethod
    def from_torch(config, model_dict,layer_id,data_type = torch.float32):
        attention = EETDistilBertAttention.from_torch(config = config, model_dict = model_dict, layer_id = layer_id,data_type = data_type)
        feedforward = EETDistilBertFeedforward.from_torch(config = config, model_dict = model_dict, layer_id = layer_id,data_type = data_type)
        layer = EETDistilBertEncoderLayer(config, attention, feedforward)
        return layer

class EETDistilBertEncoder():

    def __init__(self,EncoderLayers):
        self.layers = EncoderLayers
    def __call__(
        self,
        x,
        pre_padding_len = None,
        normalize_before = False
    ):
        for layer in self.layers:
            x = layer(x,
                      pre_padding_len = pre_padding_len,
                      normalize_before = False)
        return x
    
    @staticmethod
    def from_torch(layer_model_dict,config,layer_num,data_type = torch.float32):
        """from torch."""
        EncoderLayers = []
        for i in range(layer_num):
            if i < 10:
                EncoderLayers.extend(
                    [
                        EETDistilBertEncoderLayer.from_torch(config,layer_model_dict['layer.'+str(i)+'.'],i,data_type = data_type)
                    ]
                )
            else:
                EncoderLayers.extend(
                    [
                        EETDistilBertEncoderLayer.from_torch(config,layer_model_dict['layer.'+str(i)],i,data_type = data_type)
                    ]
                )

        eet_encoder =  EETDistilBertEncoder(EncoderLayers)
        return eet_encoder

class EETDistilBertModel():
    def __init__(self,config,embedding,encoder):
        self.embedding = embedding
        self.encoder = encoder
        self.pre_padding_len = torch.empty(0).long()
        self.position_ids = torch.arange(0,config.max_position_embeddings).reshape(1,config.max_position_embeddings).cuda()
        self.pre_padding_len = torch.empty(0).long()
        self.token_type_ids = torch.empty(0).long()
    def __call__(
        self,
        input_ids,
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

        if attention_mask is None:
            pre_padding_len = self.pre_padding_len
        else:
            # transformers 0 - padding;1 - nopadding
            pre_padding_len =  torch.sum(1 - attention_mask,1).long().cuda()
            
        embedding_out = self.embedding(input_ids,position_ids,self.token_type_ids)
        sequence_output = self.encoder(embedding_out,
                    pre_padding_len = pre_padding_len,
                    normalize_before = False)
        return sequence_output
    
    @staticmethod
    def from_pretrained(model_id_or_path: str,max_batch, data_type, device_id=0):
        """from_pretrained."""
        torch.set_grad_enabled(False)
        model_dict = {}
        embedding_dict = {}
        torch_model = DistilBertModel.from_pretrained(model_id_or_path)
        cfg = torch_model.config
        for k, v in torch_model.state_dict().items():
            if 'embeddings' in k:
                embedding_dict[k] = v
            if 'layer' in k:
                k = k[BEGIN_OF_PARAM:]
                model_dict[k] = v

        # Group by layer id in model_dict's keys
        from itertools import groupby
        layer_model_dict = {k: dict(v) for k, v in groupby(list(model_dict.items()), lambda item: item[0][:8])}


        device = "cpu" if device_id < 0 else f"cuda:{device_id}"
        activation_fn = cfg.activation
        batch_size = max_batch
        config = meta_desc(batch_size, cfg.n_heads, cfg.dim, cfg.n_layers,
                           cfg.max_position_embeddings, cfg.max_position_embeddings, data_type, device, False,
                           activation_fn)

        embedding = EETDistilBertEmbedding.from_torch(config, embedding_dict, data_type)
        # embedding = None
        encoder = EETDistilBertEncoder.from_torch(layer_model_dict, config, cfg.n_layers, data_type)
        if data_type == torch.float32:
            torch_model = torch_model.float()
        else:
            torch_model = torch_model.half()
        eet_model = EETDistilBertModel(cfg, embedding, encoder)
        return eet_model

    def from_torch(torch_model,max_batch, data_type, device_id=0):
        """from torch."""
        torch.set_grad_enabled(False)
        model_dict = {}
        embedding_dict = {}

        cfg = torch_model.config
        for k, v in torch_model.distilbert.state_dict().items():
            # print(k,v.size())
            if 'embeddings' in k:
                embedding_dict[k] = v
            if 'layer' in k:
                k = k[BEGIN_OF_PARAM:]
                model_dict[k] = v

        # Group by layer id in model_dict's keys
        from itertools import groupby
        layer_model_dict = {k: dict(v) for k, v in groupby(list(model_dict.items()), lambda item: item[0][:8])}
        
        device = "cpu" if device_id < 0 else f"cuda:{device_id}"
        activation_fn = cfg.activation
        batch_size = max_batch
        config = meta_desc(batch_size, cfg.n_heads, cfg.dim, cfg.n_layers,
                           cfg.max_position_embeddings, cfg.max_position_embeddings, data_type, device, False,
                           activation_fn)

        embedding = EETDistilBertEmbedding.from_torch(config, embedding_dict, data_type)
        # embedding = None
        encoder = EETDistilBertEncoder.from_torch(layer_model_dict, config, cfg.n_layers, data_type)
        if data_type == torch.float32:
            torch_model = torch_model.float()
        else:
            torch_model = torch_model.half()
        eet_model = EETDistilBertModel(cfg, embedding, encoder)
        return eet_model

class EETDistilBertForMaskedLM():
    def __init__(self,distilbert,activation,config,vocab_transform,vocab_layer_norm,vocab_projector):
        self.config = config
        self.distilbert = distilbert
        self.activation = activation
        self.vocab_transform = nn.Linear(config.dim, config.dim)
        self.vocab_layer_norm = nn.LayerNorm(config.dim, eps=1e-12)
        self.vocab_projector = nn.Linear(config.dim, config.vocab_size)

    def __call__(
        self,
        input_ids,
        attention_mask = None,
    ) :

        dlbrt_output = self.distilbert(
            input_ids = input_ids,
            attention_mask=attention_mask,
        )

        hidden_states = dlbrt_output  # (bs, seq_length, dim)
        prediction_logits = self.vocab_transform(hidden_states)  # (bs, seq_length, dim)
        prediction_logits = self.activation(prediction_logits)  # (bs, seq_length, dim)
        prediction_logits = self.vocab_layer_norm(prediction_logits)  # (bs, seq_length, dim)
        prediction_logits = self.vocab_projector(prediction_logits)  # (bs, seq_length, vocab_size)
        # return prediction_scores
        return MaskedLMOutput(
            loss=None,
            logits=prediction_logits,
            hidden_states=None,
            attentions=None,
        )

    def from_pretrained(model_id_or_path: str,max_batch, data_type):
        """from_pretrained"""
        torch.set_grad_enabled(False)
        model_dict = {}
        embedding_dict = {}
        torch_model = DistilBertForMaskedLM.from_pretrained(model_id_or_path)
        distilbert = EETDistilBertModel.from_torch(torch_model,max_batch,data_type)
        activation = torch_model.activation.cuda()
        vocab_transform = torch_model.activation.cuda()
        vocab_layer_norm = torch_model.activation.cuda()
        vocab_projector = torch_model.activation.cuda()

        model =  EETDistilBertForMaskedLM(distilbert, activation,torch_model.config,vocab_transform,vocab_layer_norm,vocab_projector)

        return model

class EETDistilBertForSequenceClassification():
    def __init__(self,distilbert,classifier,pre_classifier,config):
        self.config = config
        self.distilbert = distilbert
        self.classifier = classifier
        self.pre_classifier = pre_classifier

    def __call__(
        self,
        input_ids,
        attention_mask = None,
    ) :
        distilbert_output = self.distilbert(
            input_ids = input_ids,
            attention_mask=attention_mask,
        )
        hidden_state = distilbert_output  # (bs, seq_len, dim)
        pooled_output = hidden_state[:, 0]  # (bs, dim)
        pooled_output = self.pre_classifier(pooled_output)  # (bs, dim)
        pooled_output = nn.ReLU()(pooled_output)  # (bs, dim)
        logits = self.classifier(pooled_output)  # (bs, num_labels)

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
        torch_model = DistilBertForSequenceClassification.from_pretrained(model_id_or_path)
        distilbert = EETDistilBertModel.from_torch(torch_model,max_batch,data_type)
        classifier = torch_model.classifier.cuda()
        pre_classifier = torch_model.pre_classifier.cuda()

        model =  EETDistilBertForSequenceClassification(distilbert, classifier,pre_classifier,torch_model.config)

        return model

class EETDistilBertForMultipleChoice():
    def __init__(self,distilbert,classifier,pre_classifier,config):
        self.config = config
        self.distilbert = distilbert
        self.classifier = classifier
        self.pre_classifier = pre_classifier
    def __call__(
        self,
        input_ids,
        attention_mask = None,
    ) :
        num_choices = input_ids.shape[1]
        outputs = self.distilbert(
            input_ids = input_ids,
            attention_mask=attention_mask,
        )
        hidden_state = outputs  # (bs * num_choices, seq_len, dim)
        pooled_output = hidden_state[:, 0]  # (bs * num_choices, dim)
        pooled_output = self.pre_classifier(pooled_output)  # (bs * num_choices, dim)
        pooled_output = nn.ReLU()(pooled_output)  # (bs * num_choices, dim)
        logits = self.classifier(pooled_output)  # (bs * num_choices, 1)

        reshaped_logits = logits.view(-1, num_choices)  # (bs, num_choices)

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
        torch_model = DistilBertForMultipleChoice.from_pretrained(model_id_or_path)
        distilbert = EETDistilBertModel.from_torch(torch_model,max_batch,data_type)
        classifier = torch_model.classifier.cuda()
        pre_classifier = torch_model.pre_classifier.cuda()
        model =  EETDistilBertForMultipleChoice(distilbert, classifier,pre_classifier,torch_model.config)

        return model

class EETDistilBertForTokenClassification():
    def __init__(self,distilbert,classifier,config):
        self.config = config
        self.distilbert = distilbert
        self.classifier = classifier

    def __call__(
        self,
        input_ids,
        attention_mask = None,
    ) :
        sequence_output = self.distilbert(
            input_ids = input_ids,
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
        torch_model = DistilBertForTokenClassification.from_pretrained(model_id_or_path)
        distilbert = EETDistilBertModel.from_torch(torch_model,max_batch,data_type)
        classifier = torch_model.classifier.cuda()
        model =  EETDistilBertForTokenClassification(distilbert, classifier,torch_model.config)
        return model

class EETDistilBertForQuestionAnswering():
    def __init__(self,distilbert,qa_outputs,config):
        self.config = config
        self.distilbert = distilbert
        self.qa_outputs = qa_outputs

    def __call__(
        self,
        input_ids,
        attention_mask = None,
    ) :
        distilbert_output = self.distilbert(
            input_ids = input_ids,
            attention_mask=attention_mask,
        )
        hidden_states = distilbert_output  # (bs, max_query_len, dim)
        logits = self.qa_outputs(hidden_states)  # (bs, max_query_len, 2)
        start_logits, end_logits = logits.split(1, dim=-1)
        start_logits = start_logits.squeeze(-1).contiguous()  # (bs, max_query_len)
        end_logits = end_logits.squeeze(-1).contiguous()  # (bs, max_query_len)

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
        torch_model = DistilBertForQuestionAnswering.from_pretrained(model_id_or_path)
        distilbert = EETDistilBertModel.from_torch(torch_model,max_batch,data_type)
        qa_outputs = torch_model.qa_outputs.cuda()
        model =  EETDistilBertForQuestionAnswering(distilbert, qa_outputs,torch_model.config)
        return model