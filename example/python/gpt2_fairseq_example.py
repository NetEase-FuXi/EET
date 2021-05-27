import torch
import time
import numpy as np
from torch import nn
from fairseq.data.dictionary import Dictionary
from eet.fairseq.transformer import EETTransformerDecoder

using_pytorch = True
using_eet = True
using_half = False

prompt_seq_len = 4

# eet supports a maximum seq_len of 4096 
max_seq_len = 1024
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

args = Args(0, True, 512, 512, 1024, False, False, False, False, 6, 8, 2048, None, 0.1, 0.1)
embedding = nn.Embedding(13672, 512, padding_idx=1)

dictionary = Dictionary.load('../../resource/data/dict.txt')

def main():
    model_id_or_path = '../../resource/model/checkpoint_best.pt'
    torch.set_grad_enabled(False)
    pretrained_dict = torch.load(model_id_or_path)
    model_dict = {}
    tokens = np.random.randint(3,13672,max_seq_len * batch,dtype="int64")
    tokens = torch.from_numpy(tokens).long().reshape(batch, max_seq_len).cuda()
    # tokens[2:4,0:2] = 1
    if using_eet:
        data_type = torch.float32
        if using_half:
            data_type = torch.float16
        eet_config = {"data_type":data_type,"max_batch":batch,"full_seq_len":prompt_seq_len}
        eet_model = EETTransformerDecoder.from_torch(model_id_or_path = model_id_or_path,dictionary = dictionary,args = args,config = eet_config,no_encoder_attn = True)


    total_time_eet = 0
    first_pass = True
    reorder_state = None
    for step in range(prompt_seq_len-1, max_seq_len):
        print('step:',step)
        torch.cuda.synchronize()
        t1 = time.perf_counter()
        if first_pass:
            input_ids_eet = torch.clone(tokens[:, :step + 1].contiguous()).cuda().long()
        else:
            input_ids_eet = torch.clone(tokens[:, step:step + 1].contiguous()).cuda().long()
        res_eet = eet_model(input_ids_eet, reorder_state = reorder_state,first_pass = first_pass)
        torch.cuda.synchronize()
        t2 = time.perf_counter()
        print('eet time : ', t2 - t1)
        total_time_eet += (t2 - t1)

        # eet support dynamic batch according to the reorder_state
        reorder_state = torch.tensor([1,0,2,3]).cuda()

        tokens[:, : step + 1] = torch.index_select(
            tokens[:, : step + 1], dim=0, index=reorder_state
        )

        if first_pass == True:
            first_pass = False

        
    print('total time for eet : ', total_time_eet)

if __name__ == '__main__':
    main()
