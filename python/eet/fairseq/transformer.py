#
# Created by djz on 2021/01/21.
#
"""EET fairseq gpt2 model. """

import math
import time
import torch
import torch.nn as nn
import torch.nn.functional as F
from torch import Tensor
from typing import Any, Dict, List, Optional, Tuple
from fairseq.models.transformer import TransformerDecoder 
from fairseq.models.transformer import TransformerEncoder 
from fairseq.modules import  AdaptiveSoftmax
from fairseq import options
from EET import MetaDesc as meta_desc
from EET import LayerNorm as eet_layernorm
from EET import Embedding as eet_embedding
from EET import FeedForwardNetwork as eet_ffn
from EET import MaskedMultiHeadAttention as eet_attention
from EET import CrossMultiHeadAttention as eet_cross_attention
from EET import MultiHeadAttention as eet_encoder_attention

PARAM_LEN = 17
__all__ = [
    'EETTransformerLayerNorm', 'EETTransformerEmbedding', 'EETTransformerFeedforward', 'EETTransformerAttention', 
    'EETTransformerDecoderLayer', 'EETTransformerDecoder'
]

class EETTransformerLayerNorm():
    def __init__(self,args, meta_des,layernorm_weights,layernorm_bias,data_type = torch.float32):
        self.layernorm_weights = layernorm_weights.cuda().type(data_type)
        self.layernorm_bias = layernorm_bias.cuda().type(data_type)
        self.layernorm = eet_layernorm(meta_des,self.layernorm_weights,self.layernorm_bias)

    def __call__(self,
                input_ids):
        # positional_encode + embedding_lookup
        return self.layernorm.layer_norm(input_ids)
    
    @staticmethod
    def from_torch(args,meta_des,layernorm_weights,layernorm_bias,data_type = torch.float32):
        layernorm = EETTransformerLayerNorm(args,meta_des,layernorm_weights,layernorm_bias,data_type = data_type)
        return layernorm

class EETTransformerEmbedding():
    def __init__(self,args, meta_des,embedding_weights,data_type):
        self.embedding_weights = embedding_weights.cuda().type(data_type)
        self.padding_idx = 1
        self.weight = self.embedding_weights
        self.embed_scale = args.no_scale_embedding
        # print('self.embedding_weights:',self.embedding_weights,self.embedding_weights.size())
        self.embedding = eet_embedding(meta_des,self.embedding_weights,self.embedding_weights,self.embedding_weights ,self.embedding_weights ,self.embedding_weights)

    def __call__(self,
                tokens):
        # positional_encode + embedding_lookup
        return self.embedding.forward_fairseq(tokens, self.embed_scale,self.padding_idx)
    
    @staticmethod
    def from_torch(args,meta_des,embedding_weights,data_type =  torch.float32):
        feedforward = EETTransformerEmbedding(args,meta_des,embedding_weights,data_type)
        return feedforward

class EETTransformerFeedforward():
    def __init__(self,meta_des,model_dict,data_type =  torch.float32):

        self.intermediate_weights = torch.clone(torch.t([x[1] for x in model_dict.items() if 'fc1.weight' in x[0]][0]).contiguous()).contiguous().cuda().type(data_type)
        self.intermediate_bias = [x[1] for x in model_dict.items() if 'fc1.bias' in x[0]][0].cuda().type(data_type)
        self.output_weights = torch.clone(torch.t([x[1] for x in model_dict.items() if 'fc2.weight' in x[0]][0]).contiguous()).contiguous().cuda().type(data_type)
        self.output_bias = [x[1] for x in model_dict.items() if 'fc2.bias' in x[0]][0].cuda().type(data_type)
        self.layernorm_weights = [x[1] for x in model_dict.items() if 'final_layer_norm.weight' in x[0]][0].cuda().type(data_type)
        self.layernorm_bias = [x[1] for x in model_dict.items() if 'final_layer_norm.bias' in x[0]][0].cuda().type(data_type)

        self.ffn = eet_ffn(meta_des,self.intermediate_weights,self.intermediate_bias,self.output_weights,self.output_bias,self.layernorm_weights,self.layernorm_bias)

    def __call__(self,
                input_id,
                pre_layernorm = True,
                add_redusial = True):
        return self.ffn.forward(input_id,pre_layernorm,add_redusial)
        
    
    @staticmethod
    def from_torch(meta_des,model_dict, data_type = torch.float32):
        feedforward = EETTransformerFeedforward(meta_des,model_dict,data_type = data_type)
        return feedforward

class EETTransformerAttention():
    def __init__(self,meta_des, model_dict,no_encoder_attn=False,data_type = torch.float32,is_encoder = False):
        self.is_encoder = is_encoder
        if is_encoder is False:
            if no_encoder_attn is True:
                # cross_attention
                self.q_weights = torch.clone(torch.t([x[1] for x in model_dict.items() if 'self_attn.q_proj.weight' in x[0]][0]).contiguous()).cuda().type(data_type)
                self.k_weights = torch.clone(torch.t([x[1] for x in model_dict.items() if 'self_attn.k_proj.weight' in x[0]][0]).contiguous()).cuda().type(data_type)
                self.v_weights = torch.clone(torch.t([x[1] for x in model_dict.items() if 'self_attn.v_proj.weight' in x[0]][0]).contiguous()).cuda().type(data_type)
                self.q_bias = [x[1] for x in model_dict.items() if 'self_attn.q_proj.bias' in x[0]][0].cuda().type(data_type)
                self.k_bias = [x[1] for x in model_dict.items() if 'self_attn.k_proj.bias' in x[0]][0].cuda().type(data_type)
                self.v_bias = [x[1] for x in model_dict.items() if 'self_attn.v_proj.bias' in x[0]][0].cuda().type(data_type)
                self.out_weights = torch.clone(torch.t([x[1] for x in model_dict.items() if 'self_attn.out_proj.weight' in x[0]][0]).contiguous()).cuda().type(data_type)
                self.out_bias = [x[1] for x in model_dict.items() if 'self_attn.out_proj.bias' in x[0]][0].cuda().type(data_type)
                self.layernorm_weights = [x[1] for x in model_dict.items() if 'self_attn_layer_norm.weight' in x[0]][0].cuda().type(data_type)
                self.layernorm_bias = [x[1] for x in model_dict.items() if 'self_attn_layer_norm.bias' in x[0]][0].cuda().type(data_type)
                self.attention = eet_attention(meta_des,self.q_weights,self.k_weights,self.v_weights,self.q_bias,self.k_bias,self.v_bias,self.out_weights,self.out_bias,self.layernorm_weights,self.layernorm_bias)

            else:
                self.q_weights = torch.clone(torch.t([x[1] for x in model_dict.items() if 'encoder_attn.q_proj.weight' in x[0]][0]).contiguous()).cuda()
                self.k_weights = torch.clone(torch.t([x[1] for x in model_dict.items() if 'encoder_attn.k_proj.weight' in x[0]][0]).contiguous()).cuda()
                self.v_weights = torch.clone(torch.t([x[1] for x in model_dict.items() if 'encoder_attn.v_proj.weight' in x[0]][0]).contiguous()).cuda()
                self.q_bias = [x[1] for x in model_dict.items() if 'encoder_attn.q_proj.bias' in x[0]][0].cuda()
                self.k_bias = [x[1] for x in model_dict.items() if 'encoder_attn.k_proj.bias' in x[0]][0].cuda()
                self.v_bias = [x[1] for x in model_dict.items() if 'encoder_attn.v_proj.bias' in x[0]][0].cuda()
                self.out_weights = torch.clone(torch.t([x[1] for x in model_dict.items() if 'encoder_attn.out_proj.weight' in x[0]][0]).contiguous()).cuda()
                self.out_bias = [x[1] for x in model_dict.items() if 'encoder_attn.out_proj.bias' in x[0]][0].cuda()
                self.layernorm_weights = [x[1] for x in model_dict.items() if 'encoder_attn_layer_norm.weight' in x[0]][0].cuda()
                self.layernorm_bias = [x[1] for x in model_dict.items() if 'encoder_attn_layer_norm.bias' in x[0]][0].cuda()
                self.attention = eet_cross_attention(meta_des,self.q_weights,self.k_weights,self.v_weights,self.q_bias,self.k_bias,self.v_bias,self.out_weights,self.out_bias,self.layernorm_weights,self.layernorm_bias)
        else:
            # transformer encoder
            self.q_weights = torch.clone(torch.t([x[1] for x in model_dict.items() if 'self_attn.q_proj.weight' in x[0]][0]).contiguous()).cuda().type(data_type)
            self.k_weights = torch.clone(torch.t([x[1] for x in model_dict.items() if 'self_attn.k_proj.weight' in x[0]][0]).contiguous()).cuda().type(data_type)
            self.v_weights = torch.clone(torch.t([x[1] for x in model_dict.items() if 'self_attn.v_proj.weight' in x[0]][0]).contiguous()).cuda().type(data_type)
            self.q_bias = [x[1] for x in model_dict.items() if 'self_attn.q_proj.bias' in x[0]][0].cuda().type(data_type)
            self.k_bias = [x[1] for x in model_dict.items() if 'self_attn.k_proj.bias' in x[0]][0].cuda().type(data_type)
            self.v_bias = [x[1] for x in model_dict.items() if 'self_attn.v_proj.bias' in x[0]][0].cuda().type(data_type)
            self.out_weights = torch.clone(torch.t([x[1] for x in model_dict.items() if 'self_attn.out_proj.weight' in x[0]][0]).contiguous()).cuda().type(data_type)
            self.out_bias = [x[1] for x in model_dict.items() if 'self_attn.out_proj.bias' in x[0]][0].cuda().type(data_type)
            self.layernorm_weights = [x[1] for x in model_dict.items() if 'self_attn_layer_norm.weight' in x[0]][0].cuda().type(data_type)
            self.layernorm_bias = [x[1] for x in model_dict.items() if 'self_attn_layer_norm.bias' in x[0]][0].cuda().type(data_type)
            self.attention = eet_encoder_attention(meta_des,self.q_weights,self.k_weights,self.v_weights,self.q_bias,self.k_bias,self.v_bias,self.out_weights,self.out_bias,self.layernorm_weights,self.layernorm_bias)

    def __call__(self,
                input_id,
                padding_index,
                encoder_out = None,
                encoder_padding_mask = None,
                pre_layernorm = True,
                add_redusial = True,
                first_pass = False):

        if self.is_encoder:
            return self.attention.forward(input_id,padding_index,pre_layernorm,add_redusial)
        else:
            if encoder_out is None:
                # self_atten
                return self.attention.forward(input_id,padding_index,pre_layernorm,add_redusial,first_pass)
            else:
                # cross_atten
                return self.attention.forward(input_id,encoder_out,padding_index,pre_layernorm,add_redusial,encoder_padding_mask,first_pass)

    @staticmethod
    def from_torch(meta_des,model_dict, no_encoder_attn=True,data_type = torch.float32,is_encoder = False):
        attention = EETTransformerAttention(meta_des,model_dict,no_encoder_attn,data_type = data_type,is_encoder = is_encoder)
        return attention

class EETTransformerDecoderLayer():
    def __init__(self,args, attention,feedforward,cross_attention = None, no_encoder_attn=False):
        self.args = args
        self.attetion = attention
        self.feedforward = feedforward
        if no_encoder_attn == False:
            self.cross_attention = cross_attention
        self.pre_layernorm = args.decoder_normalize_before


        self.add_redusial = True

    def __call__(self,
                x,
                attention_padding_mask = None,
                encoder_out = None,
                encoder_padding_mask  = None,
                first_pass = False):
        if encoder_out is not None:
            ''' self_attn -> cross_attn -> ffn'''
            self_attn_out = self.attetion(input_id = x,
                        padding_index = attention_padding_mask,
                        pre_layernorm = self.pre_layernorm,
                        add_redusial = self.add_redusial,
                        first_pass = first_pass)
            cross_attn_out = self.cross_attention(input_id = self_attn_out,
                        padding_index = attention_padding_mask,
                        encoder_out = encoder_out,
                        encoder_padding_mask = encoder_padding_mask,
                        pre_layernorm = self.pre_layernorm,
                        add_redusial = self.add_redusial,
                        first_pass = first_pass)
            out = self.feedforward(cross_attn_out,
                        pre_layernorm = self.pre_layernorm,
                        add_redusial = self.add_redusial)
        else:
            ''' self_attn -> ffn'''
            self_attn_out = self.attetion(input_id = x,
                        padding_index = attention_padding_mask,
                        pre_layernorm = self.pre_layernorm,
                        add_redusial = self.add_redusial,
                        first_pass = first_pass)
            out = self.feedforward(self_attn_out,
                        pre_layernorm = self.pre_layernorm,
                        add_redusial = self.add_redusial)
        return out

    @staticmethod
    def from_torch(args, meta_des, model_dict,no_encoder_attn=False,data_type = torch.float32):
        attention = EETTransformerAttention.from_torch(meta_des = meta_des, model_dict = model_dict,data_type = data_type,is_encoder = False)
        feedforward = EETTransformerFeedforward.from_torch(meta_des = meta_des, model_dict = model_dict,data_type = data_type)
        if no_encoder_attn == False:
            cross_attention = EETTransformerAttention.from_torch(meta_des = meta_des, model_dict = model_dict,no_encoder_attn = no_encoder_attn,data_type = data_type,is_encoder = True)
            layer = EETTransformerDecoderLayer(args, attention, feedforward,cross_attention,no_encoder_attn)
        else:
            layer = EETTransformerDecoderLayer(args, attention, feedforward,no_encoder_attn)
        return layer

class EETTransformerDecoder():
    """
    EETTransformerDecoder consisting of layers. Each layer
    is a :class:`EETTransformerDecoderLayer`.

    Args:
        args (argparse.Namespace): parsed command-line arguments
        dictionary (~fairseq.data.Dictionary): decoding dictionary
        embed_tokens (torch.nn.Embedding): output embedding
        embedding(EETTransformerEmbedding) : class: 'EETTransformerEmbedding'
        DecoderLayers(EETTransformerDecoderLayer) class: 'EETTransformerDecoderLayer'
        layer_norm(EETTransformerLayerNorm) class: 'EETTransformerLayerNorm'
        no_encoder_attn (bool, optional): whether to attend to encoder outputs
            (default: False).
    """
    def __init__(self, args, max_batch, dictionary, embed_tokens, DecoderLayers, layer_norm):
        self.layers = DecoderLayers
        self.layer_norm = layer_norm
        self.cross_self_attention = False
        self.adaptive_softmax = None
        self.share_input_output_embed = args.share_decoder_input_output_embed
        self.output_embed_dim = args.decoder_output_dim
        self.embed_tokens = embed_tokens
        self.self_attn_padding_mask = torch.empty(0)

        if args.adaptive_softmax_cutoff is not None:
            self.adaptive_softmax = AdaptiveSoftmax(
                len(dictionary),
                self.output_embed_dim,
                options.eval_str_list(args.adaptive_softmax_cutoff, type=int),
                dropout=args.adaptive_softmax_dropout,
                adaptive_inputs=embed_tokens if args.tie_adaptive_weights else None,
                factor=args.adaptive_softmax_factor,
                tie_proj=args.tie_adaptive_proj,
            )
        elif not self.share_input_output_embed:
            self.embed_out = nn.Parameter(
                torch.Tensor(len(dictionary), self.output_embed_dim)
            )
            nn.init.normal_(self.embed_out, mean=0, std=self.output_embed_dim ** -0.5)

    def __call__(
        self,
        prev_output_tokens,
        encoder_out: Optional[Dict[str, List[Tensor]]] = None,
        features_only: bool = False,
        first_pass = False
    ):
        """
        Args:
            prev_output_tokens : (LongTensor): previous decoder outputs of shape
                `(batch, tgt_len)`, for teacher forcing
            encoder_out (optional): output from the encoder, used for
                encoder-side attention
            features_only (bool, optional): only return features without
                applying output layer (default: False).
            first_pass:full decoder or incremental decoder,first step is full decoder,other is incremental decoder.
        Returns:
            the decoder's output of shape `(batch, tgt_len, vocab)`
        """
        # print('prev_output_tokens-----:',prev_output_tokens)
        x = self.embed_tokens(prev_output_tokens)

        """ EET will process self_attn_padding_mask to surport EET """
        self_attn_padding_mask: Optional[Tensor] = None
        if self.cross_self_attention or prev_output_tokens.eq(self.embed_tokens.padding_idx).any():
            self_attn_padding_mask = prev_output_tokens.eq(self.embed_tokens.padding_idx)

        if (encoder_out is not None and len(encoder_out["encoder_padding_mask"]) > 0):
            encoder_padding_mask = encoder_out["encoder_padding_mask"][0]
        else:
            encoder_padding_mask = None
 
        if (encoder_out is not None and len(encoder_out["encoder_out"]) > 0):
            encoder_out = encoder_out["encoder_out"][0]
        else:
            encoder_out = None
        if self_attn_padding_mask is None:
            self_attn_padding_mask = self.self_attn_padding_mask

        for layer in self.layers:
            x = layer(x,
                    attention_padding_mask = self_attn_padding_mask,
                    encoder_out = encoder_out,
                    encoder_padding_mask = encoder_padding_mask,
                    first_pass = first_pass)
        
        x = self.layer_norm(x)
        if not features_only:
            x = self.output_layer(x)
        
        return x

    def output_layer(self, features):
        """Project features to the vocabulary size."""
        if self.adaptive_softmax is None:
            if self.share_input_output_embed:
                return F.linear(features, self.embed_tokens.weight)
            else:

                return F.linear(features, self.embed_out)
        else:
            return features

    @staticmethod
    def from_torch(model_id_or_path: str, dictionary, args, config:dict, no_encoder_attn=False):
        """from torch."""
        """
        Args:
            model_id_or_path : pytorch model path
            dictionary (~fairseq.data.Dictionary): decoding dictionary
            args (argparse.Namespace): parsed command-line arguments
            config:dic[max_full_seq_len,max_batch,data_type]
            {   
                full_seq_len: The maximum length that can be supported by full decoding 
                max_batch: the largest batch_size that can be supported, and it is supported if it is smaller than max_batch, so as to support dynamic batch
                data_type: data_type (default: torch.float32)
            }
            no_encoder_attn (bool, optional): whether to attend to encoder outputs
                (default: False).
        Returns:
            eet_decoder : EETTransformerDecoder
        """

        torch.set_grad_enabled(False)
        pretrained_dict = torch.load(model_id_or_path)

        full_seq_len = config['full_seq_len']
        batch_size = config['max_batch']
        data_type = config['data_type']

        model_dict = {}
        DecoderLayers = []
        for k, v in pretrained_dict['model'].items():
            model_dict[k] = v
        from itertools import groupby
        # Intercept k,num = length of 'decoder.layers.**'=17; If your weight name has changed please change it here
        layer_model_dict = {k: dict(v) for k, v in groupby(list(model_dict.items()), lambda item: item[0][:PARAM_LEN])}

        device = "cuda:0"
        activation_fn = args.activation_fn
        meta_des = meta_desc(batch_size, args.decoder_attention_heads, args.decoder_embed_dim, args.decoder_layers ,128, full_seq_len, data_type, device, False, activation_fn)
        embedding = EETTransformerEmbedding.from_torch(args,meta_des,model_dict['decoder.embed_tokens.weight'],data_type)
        layer_norm = EETTransformerLayerNorm.from_torch(args,meta_des,model_dict['decoder.layer_norm.weight'],model_dict['decoder.layer_norm.bias'],data_type)
        # layer_norm = None
        for i in range(args.decoder_layers):
            if i < 10:
                DecoderLayers.extend(
                    [
                        EETTransformerDecoderLayer.from_torch(args,meta_des,layer_model_dict['decoder.layers.'+str(i)+'.'],no_encoder_attn,data_type)
                    ]
                )
            else:
                DecoderLayers.extend(
                    [
                        EETTransformerDecoderLayer.from_torch(args,meta_des,layer_model_dict['decoder.layers.'+str(i)],no_encoder_attn,data_type)
                    ]
                )

        eet_decoder = EETTransformerDecoder(args, batch_size, dictionary, embedding, DecoderLayers,layer_norm)

        return eet_decoder
