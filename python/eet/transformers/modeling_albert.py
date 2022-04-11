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
from EET import MetaDesc as meta_desc
from EET import FeedForwardNetwork as eet_ffn
from EET import MultiHeadAttention as eet_attention
from EET import Embedding as eet_embedding

BEGIN_OF_PARAM = 8
BEGIN_OF_ALBERT = 7

__all__ = [
    'EETAlbertEmbedding', 'EETAlbertFeedforward', 'EETAlbertAttention', 'EETAlbertEncoderLayer', 'EETAlbertEncoder', 'EETAlbertModel',
    'EETAlbertForPreTraining','EETAlbertForMaskedLM','EETAlbertForSequenceClassification','EETAlbertForTokenClassification','EETAlbertForQuestionAnswering',
    'EETAlbertForMultipleChoice',
]

@dataclass
class AlbertForPreTrainingOutput(ModelOutput):
    loss: Optional[torch.FloatTensor] = None
    prediction_logits: torch.FloatTensor = None
    sop_logits: torch.FloatTensor = None
    hidden_states: Optional[Tuple[torch.FloatTensor]] = None
    attentions: Optional[Tuple[torch.FloatTensor]] = None



class EETAlbertEmbedding():
    def __init__(self,config,embedding_dict,data_type = torch.float32):
        self.if_layernorm = True
        self.embedding_weights = embedding_dict['embeddings.word_embeddings.weight'].cuda().type(data_type)
        self.position_weights = embedding_dict['embeddings.position_embeddings.weight'].cuda().type(data_type)
        self.token_type_weights = embedding_dict['embeddings.token_type_embeddings.weight'].cuda().type(data_type)
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
        feedforward = EETAlbertEmbedding(config,embedding_dict,data_type = data_type)
        return feedforward

class EETAlbertFeedforward():
    def __init__(self,config,model_dict,layer_id,data_type = torch.float32):  
        self.intermediate_weights = torch.t([x[1] for x in model_dict.items() if 'ffn.weight' in x[0]][0]).contiguous().contiguous().cuda().type(data_type)
        self.intermediate_bias = [x[1] for x in model_dict.items() if 'ffn.bias' in x[0]][0].cuda().type(data_type)
        self.output_weights = torch.t([x[1] for x in model_dict.items() if 'ffn_output.weight' in x[0]][0]).contiguous().contiguous().cuda().type(data_type)
        self.output_bias = [x[1] for x in model_dict.items() if 'ffn_output.bias' in x[0]][0].contiguous().cuda().type(data_type)
        self.layernorm_weights = [x[1] for x in model_dict.items() if '.full_layer_layer_norm.weight' in x[0]][0].cuda().type(data_type)
        self.layernorm_bias = [x[1] for x in model_dict.items() if '.full_layer_layer_norm.bias' in x[0]][0].cuda().type(data_type)
       

        self.ffn = eet_ffn(config,self.intermediate_weights,self.intermediate_bias,self.output_weights,self.output_bias,self.layernorm_weights,self.layernorm_bias)
    def __call__(self, input_id, pre_layernorm = True, add_redusial = True):
        res = self.ffn.forward(input_id,pre_layernorm,add_redusial)
        return res
    
    def from_torch(config,model_dict,layer_id,data_type = torch.float32):
        feedforward = EETAlbertFeedforward(config,model_dict,layer_id,data_type = data_type)
        return feedforward

class FCLayer():
    def __init__(self, config, model_dict,data_type=torch.float32):
        self.w =  model_dict['embeddin']['embedding_hidden_mapping_in.weight'].contiguous().cuda().type(data_type)
        self.b =  model_dict['embeddin']['embedding_hidden_mapping_in.bias'].contiguous().cuda().type(data_type)
    def __call__(self,input):
        return F.linear(input,weight=self.w, bias=self.b)

class EETAlbertAttention():
    def __init__(self,config, model_dict,layer_id,data_type = torch.float32):
        q_weights = [x[1] for x in model_dict.items() if 'query.weight' in x[0]][0].contiguous().cuda().type(data_type)
        k_weights = [x[1] for x in model_dict.items() if 'key.weight' in x[0]][0].contiguous().cuda().type(data_type)
        v_weights = [x[1] for x in model_dict.items() if 'value.weight' in x[0]][0].contiguous().cuda().type(data_type)
        self.qkv_weight = torch.cat((q_weights,k_weights,v_weights),0).transpose(0,1).contiguous()

        self.q_bias = [x[1] for x in model_dict.items() if 'query.bias' in x[0]][0].cuda().type(data_type)
        self.k_bias = [x[1] for x in model_dict.items() if 'key.bias' in x[0]][0].cuda().type(data_type)
        self.v_bias = [x[1] for x in model_dict.items() if 'value.bias' in x[0]][0].cuda().type(data_type)
        self.out_weights = torch.t([x[1] for x in model_dict.items() if 'attention.dense.weight' in x[0]][0]).contiguous().cuda().type(data_type)
        self.out_bias = [x[1] for x in model_dict.items() if 'attention.dense.bias' in x[0]][0].cuda().type(data_type)
        self.layernorm_weights = [x[1] for x in model_dict.items() if 'attention.LayerNorm.weight' in x[0]][0].cuda().type(data_type)
        self.layernorm_bias = [x[1] for x in model_dict.items() if 'attention.LayerNorm.bias' in x[0]][0].cuda().type(data_type)

        self.attention = eet_attention(config,self.qkv_weight,self.q_bias,self.k_bias,self.v_bias,self.out_weights,self.out_bias,self.layernorm_weights,self.layernorm_bias)

    def __call__(self,
                input_id,
                pre_padding_len,
                pre_layernorm = False,
                add_redusial = True,
                need_sequence_mask = False):
        return self.attention.forward(input_id,pre_padding_len,pre_layernorm,add_redusial,need_sequence_mask)

    @staticmethod
    def from_torch(config,model_dict,layer_id,data_type = torch.float32):
        attention = EETAlbertAttention(config,model_dict,layer_id,data_type = data_type)
        return attention

class EETAlbertEncoderLayer():
    def __init__(self, config, attention,feedforward):
        self.attetion = attention
        self.feedforward = feedforward

    def __call__(self,
                x,
                pre_padding_len = None,
                normalize_before = False):

        ''' albert model struct '''
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
        attention = EETAlbertAttention.from_torch(config = config, model_dict = model_dict, layer_id = layer_id,data_type = data_type)
        feedforward = EETAlbertFeedforward.from_torch(config = config, model_dict = model_dict, layer_id = layer_id,data_type = data_type)
        layer = EETAlbertEncoderLayer(config, attention, feedforward)
        return layer

class EETAlbertEncoder():
    def __init__(self,EncoderLayers, hidden_mapping_in, config,cfg):
        self.hidden_mapping_in = hidden_mapping_in
        self.layers = EncoderLayers
        self.config = config
        self.cfg = cfg

    def __call__(
        self,
        x,
        pre_padding_len = None,
        normalize_before = False,

    ):
        x = self.hidden_mapping_in(x)
        for i in range(self.cfg.num_hidden_layers):
            x = self.layers[0](x,
                      pre_padding_len = pre_padding_len,
                      normalize_before = False)
        return x
    
    @staticmethod
    def from_torch(layer_model_dict,config,cfg,data_type = torch.float32):
        """from torch."""

        
        hidden_mapping_in =  FCLayer(cfg, layer_model_dict, data_type)

        EncoderLayers = []
        EncoderLayers.extend( [EETAlbertEncoderLayer.from_torch(config,layer_model_dict['albert_l'], 0, data_type = data_type)]  )


        eet_encoder =  EETAlbertEncoder(EncoderLayers, hidden_mapping_in, config,cfg)
        return eet_encoder

class EETAlbertModel():
    def __init__(self,config, embedding,encoder,pooler_activation,pooler):
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

        sequence_output = self.encoder(embedding_out,
                    pre_padding_len = pre_padding_len,
                    normalize_before = False)

        pooled_output = self.pooler_activation(self.pooler(sequence_output[:, 0])) if self.pooler is not None else None

        return sequence_output,pooled_output

    @staticmethod
    def from_pretrained(model_id_or_path: str,max_batch, data_type):
        """from_pretrained."""
    
        torch.set_grad_enabled(False)
        model_dict = {}
        embedding_dict = {}
        torch_model = AlbertModel.from_pretrained(model_id_or_path)
        cfg = torch_model.config

        for k, v in torch_model.state_dict().items():
            if 'embeddings' in k:
                embedding_dict[k] = v
            if ('layer' in k) or ('hidden_mapping' in k):
                #BEGIN_OF_PARAM(Length of the beginning of the parameter):
                #like 'encoder.layer.0.attention.self.query.weight',the BEGIN_OF_PARAM is the length of 'encoder.'-->8
                k = k[BEGIN_OF_PARAM:]
                model_dict[k] = v

        from itertools import groupby
        layer_model_dict = {k: dict(v) for k, v in groupby(list(model_dict.items()), lambda item: item[0][:BEGIN_OF_PARAM])}

        device = "cuda:0"
        activation_fn = cfg.hidden_act
        batch_size = max_batch
        config = meta_desc(batch_size, cfg.num_attention_heads, cfg.hidden_size, cfg.num_hidden_layers , cfg.max_position_embeddings, cfg.max_position_embeddings, data_type, device, False, activation_fn)
        config_emb = meta_desc(batch_size, cfg.num_attention_heads, cfg.embedding_size, cfg.num_hidden_layers , cfg.max_position_embeddings, cfg.max_position_embeddings, data_type, device, False, activation_fn)

        embedding = EETAlbertEmbedding.from_torch(config_emb,embedding_dict,data_type)
        # embedding = None
        encoder = EETAlbertEncoder.from_torch(layer_model_dict,config, cfg, data_type)
        if data_type == torch.float32:
            torch_model = torch_model.float()
        else:
            torch_model = torch_model.half()
        eet_model =  EETAlbertModel(cfg,embedding, encoder,torch_model.pooler_activation,torch_model.pooler)
        return eet_model

    def from_torch(torch_model,max_batch, data_type):
        """from torch."""
        torch.set_grad_enabled(False)
        model_dict = {}
        embedding_dict = {}
        cfg = torch_model.config
        for k, v in torch_model.state_dict().items():
            if 'embeddings' in k:
                k = k[BEGIN_OF_ALBERT:]
                embedding_dict[k] = v
            if ('layer' in k) or ('hidden_mapping' in k):
                #BEGIN_OF_PARAM(Length of the beginning of the parameter):
                #like 'encoder.layer.0.attention.self.query.weight',the BEGIN_OF_PARAM is the length of 'encoder.'-->8
                k = k[BEGIN_OF_PARAM + BEGIN_OF_ALBERT:]
                model_dict[k] = v

        from itertools import groupby
        layer_model_dict = {k: dict(v) for k, v in groupby(list(model_dict.items()), lambda item: item[0][:BEGIN_OF_PARAM])}

        device = "cuda:0"
        activation_fn = cfg.hidden_act
        batch_size = max_batch
        config = meta_desc(batch_size, cfg.num_attention_heads, cfg.hidden_size, cfg.num_hidden_layers , cfg.max_position_embeddings, cfg.max_position_embeddings, data_type, device, False, activation_fn)
        config_emb = meta_desc(batch_size, cfg.num_attention_heads, cfg.embedding_size, cfg.num_hidden_layers , cfg.max_position_embeddings, cfg.max_position_embeddings, data_type, device, False, activation_fn)

        embedding = EETAlbertEmbedding.from_torch(config_emb,embedding_dict,data_type)
        # embedding = None
        encoder = EETAlbertEncoder.from_torch(layer_model_dict,config, cfg, data_type)
        if data_type == torch.float32:
            torch_model = torch_model.float()
        else:
            torch_model = torch_model.half()

        eet_model =  EETAlbertModel(cfg,embedding, encoder,torch_model.albert.pooler_activation,torch_model.albert.pooler)
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