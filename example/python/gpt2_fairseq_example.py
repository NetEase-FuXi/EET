import torch
from torch import nn
from fairseq.data.dictionary import Dictionary
from eet.fairseq.transformer import EETTransformerDecoder

using_half = False
full_seq_len = 512 # prompt length
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

args = Args(0, True, 1280, 1280, 1024, False, False, False, True, 36, 20, 5120, None, 0.1, 0.1)

vocab_size = 13672
embedding = nn.Embedding(vocab_size, 1280, padding_idx=1)

dictionary = Dictionary.load('resource/dict.txt')

def main():
    model_id_or_path = 'python/resource/model/checkpoint_best.pt'
    torch.set_grad_enabled(False)

    # prompt context for full docoder
    inputs = np.random.randint(0,vocab_size,1 * batch,dtype="int64")
    input_full_decoder = torch.from_numpy(input).long().reshape(batch, seq_len).cuda()

    # fake prediction results for incremental decoder
    input = np.random.randint(0,vocab_size,seq_len * batch,dtype="int64")
    input_inc_decoder = torch.from_numpy(inputs).long().reshape(batch, 1).cuda()


    data_type = torch.float32
    embedding.cuda()
    if using_half:
        data_type = torch.float16
        embedding.half()
    eet_config = {"data_type":data_type,"embed_tokens":embedding,"max_batch":batch,"full_seq_len":full_seq_len} 
    eet_model = EETTransformerDecoder.from_torch(model_id_or_path = model_id_or_path,dictionary = dictionary,args = args,config = eet_config,no_encoder_attn = True)

    input_ids = input_full_decoder
    first_pass = True

    for i in range(100):
        print('i--:',i)
        res_eet = eet_model(input_ids, first_pass = first_pass)
        input_ids = input_inc_decoder
        if first_pass == True:
            first_pass = False
if __name__ == '__main__':
    main()
