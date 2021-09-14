import torch
import time
import numpy as np
from torch import nn
from fairseq.data.dictionary import Dictionary
from fairseq.models.transformer import TransformerDecoder 
from eet.fairseq.transformer import EETTransformerDecoder
import sys

context_len = 512
batch = 4
max_seq_len = 1024

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

#1280 -- hidden_units
#16   -- layer num
args = Args(0, True, 1024, 1024, max_seq_len, False, False, False, True, 24, 16, 1024 * 4, None, 0.1, 0.1)
embedding = nn.Embedding(13672, 1024, padding_idx=1)

dictionary = Dictionary.load('resource/dict.txt')

def main():
    torch.set_grad_enabled(False)

    tokens = np.random.randint(3,13672,max_seq_len * batch,dtype="int64")
    tokens = torch.from_numpy(tokens).long().reshape(batch, max_seq_len).cuda()

    torch_decoder = TransformerDecoder(args, dictionary, embedding, True).cuda().half().eval()

    eet_config = {"data_type":torch.float16,"max_batch":batch,"full_seq_len":context_len}
    eet_model = EETTransformerDecoder.from_buffer(torch_decoder = torch_decoder,dictionary = dictionary,args = args,config = eet_config,no_encoder_attn = True)

    torch.cuda.synchronize()
    t1 = time.perf_counter()

    for i in range(100):
        first_pass = True
        reorder_state = None
        for step in range(context_len - 1, max_seq_len):
            if first_pass:
                input_ids_eet = tokens[:, :step + 1].contiguous().cuda().long()
            else:
                input_ids_eet = tokens[:, step:step + 1].contiguous().cuda().long()
            res_eet = eet_model(input_ids_eet,reorder_state=reorder_state, first_pass = first_pass)
            first_pass = False

    torch.cuda.synchronize()
    t2 = time.perf_counter()
    print('Time for EET : ', t2 - t1)

    torch.cuda.synchronize()
    t3 = time.perf_counter()

    for i in range(100):
        incremental_state = {}
        for step in range(0, max_seq_len):
            res_torch, incremental_state = torch_decoder(tokens[:,:step+1], incremental_state=incremental_state)

    torch.cuda.synchronize()
    t4 = time.perf_counter()
    print('Time for Fairseq : ', t4 - t3)
    print('SpeedUp is : ', (t4 - t3)/(t2- t1))


if __name__ == '__main__':
    main()
