import torch
import time
import numpy as np
from torch import nn
from fairseq.data.dictionary import Dictionary
from fairseq.models.transformer import TransformerDecoder 
from eet.fairseq.transformer import EETTransformerDecoder
import sys

using_eet = True
using_half = False
seq_len = 102
batch = 4

class Args(object):
    def __init__(self,
                 decoder_layerdrop,
                 share_decoder_input_output_embed,
                 decoder_embed_dim,
                 decoder_output_dim,
                 max_target_positions,
                 no_scale_embedding,
                 decoder_learned_pos,
                 no_token_positional_embeddings,
                 decoder_normalize_before,
                 decoder_layers,
                 decoder_attention_heads,
                 decoder_ffn_embed_dim,
                 adaptive_softmax_cutoff=None,
                 dropout=0.1,
                 attention_dropout=0.1,
                 activation_fn='relu',
                 adaptive_input=False,
                 quant_noise_pq=0
                 ):
        super().__init__()
        self.decoder_layerdrop = decoder_layerdrop
        self.share_decoder_input_output_embed = share_decoder_input_output_embed
        self.decoder_embed_dim = decoder_embed_dim
        self.decoder_output_dim = decoder_output_dim
        self.max_target_positions = max_target_positions
        self.no_scale_embedding = no_scale_embedding
        self.decoder_learned_pos = decoder_learned_pos
        self.no_token_positional_embeddings = no_token_positional_embeddings
        self.decoder_normalize_before = decoder_normalize_before
        self.decoder_layers = decoder_layers
        self.decoder_attention_heads = decoder_attention_heads
        self.decoder_ffn_embed_dim = decoder_ffn_embed_dim
        self.adaptive_softmax_cutoff = adaptive_softmax_cutoff
        self.dropout = dropout
        self.attention_dropout = attention_dropout
        self.activation_fn = activation_fn
        self.fp16 = False
        self.adaptive_input = adaptive_input
        self.quant_noise_pq = quant_noise_pq

        assert self.decoder_embed_dim == self.decoder_output_dim

# args = Args(0, True, 1280, 1280, 1024, False, False, False, True, 1, 20, 5120, None, 0.1, 0.1)
# embedding = nn.Embedding(13672, 1280, padding_idx=1)

# args = Args(0, True, 4, 4, 1024, False, False, False, True, 1, 1, 16, None, 0.1, 0.1, 'gelu')
# embedding = nn.Embedding(13672, 4, padding_idx=1)
args = Args(0, True, 1024, 1024, 1024, False, False, False, True, 16, 16, 4096, None, 0.1, 0.1)
embedding = nn.Embedding(13672, 1024, padding_idx=1)

# args = Args(0, True,  12288,  12288, 1024, False, False, False, True, 1, 96, 49152, None, 0.1, 0.1)
# embedding = nn.Embedding(13672, 12288, padding_idx=1)

# args = Args(0, True, 768, 768, 768, False, False, False, True, 12, 12, 3072, None, 0.1, 0.1)
# embedding = nn.Embedding(13672, 768, padding_idx=1)

# args = Args(0, True, 16, 16, 1024, False, False, False, True, 4, 4, 64, None, 0.1, 0.1)
# embedding = nn.Embedding(13672, 16, padding_idx=1)
# args = Args(0, True, 16, 16, 1024, False, False, False, True, 4, 4, 64, None, 0.1, 0.1)
# embedding = nn.Embedding(13672, 16, padding_idx=1)
dictionary = Dictionary.load('resource/data/dict.txt')

def main():
    torch.set_grad_enabled(False)

    input = np.random.randint(3,9000,seq_len * batch,dtype="int64")
    inputs = np.random.randint(3,9000,1 * batch,dtype="int64")

    input_full = torch.from_numpy(input).long().reshape(batch, seq_len).cuda()
    input_inc = torch.from_numpy(inputs).long().reshape(batch, 1).cuda()


    data_type = torch.float32
    if using_half:
        data_type = torch.float16
    torch_decoder = TransformerDecoder(args, dictionary, embedding, True)
    if using_half:
        torch_decoder.half()
    torch_decoder.cuda().eval()
    eet_config = {"data_type":data_type,"max_batch":batch,"full_seq_len":seq_len}
    eet_model = EETTransformerDecoder.from_buffer(torch_decoder = torch_decoder,dictionary = dictionary,args = args,config = eet_config,no_encoder_attn = True)

    total_time_ft = 0
    full_decoder_time_ft = 0
    inc_decoder_time_ft = 0

    input_ids = input_full
    first_pass = True

    for i in range(1024 - seq_len):
        torch.cuda.synchronize()
        t1 = time.perf_counter()
        
        res_ft = eet_model(input_ids, first_pass = first_pass)
        torch.cuda.synchronize()
        t2 = time.perf_counter()
        if i == 0:
            full_decoder_time_ft = (t2 - t1)
        else:
            inc_decoder_time_ft += (t2 - t1)
        print('ft time : ',i, t2 - t1)
        total_time_ft += (t2 - t1)
        # print('res_ft:',res_ft)
        input_ids = input_inc
        if first_pass == True:
            first_pass = False

    print('full_decoder time for ft : ', full_decoder_time_ft)
    print('inc_decoder time for ft : ', inc_decoder_time_ft)
    print('total time for ft : ', total_time_ft)

if __name__ == '__main__':
    main()
