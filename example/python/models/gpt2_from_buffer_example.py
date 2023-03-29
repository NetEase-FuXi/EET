from pyexpat import model
import torch
import time
import numpy as np
from torch import nn
from fairseq.data.dictionary import Dictionary
from fairseq.models.transformer import TransformerDecoder 
from eet.fairseq.transformer import EETTransformerDecoder
import sys
import argparse

context_len = 512
batch = 4
max_seq_len = 1024
loop = 1

args = argparse.Namespace()
args.decoder_layerdrop = 0
args.share_decoder_input_output_embed = True
args.decoder_embed_dim = 768
args.decoder_output_dim = 768
args.max_target_positions = context_len
args.no_scale_embedding = False
args.decoder_learned_pos = False
args.no_token_positional_embeddings = False
args.decoder_normalize_before = True
args.decoder_layers = 12
args.decoder_attention_heads = 12
args.decoder_ffn_embed_dim = 768 * 4
args.adaptive_softmax_cutoff = None
args.dropout = 0.1
args.attention_dropout = 0.1
args.activation_fn = 'relu'
args.fp16 = False
args.adaptive_input = False
args.quant_noise_pq = 0

#768 -- hidden_units
#12   -- layer num
embedding = nn.Embedding(13672, 768, padding_idx=1)

dictionary = Dictionary.load('../resource/dict.txt')

def main():
    '''
    此demo是用于方便测试性能对比，直接构造模型，随机生成权重参数。
    '''
    torch.set_grad_enabled(False)
    tokens = np.random.randint(3, 13672, max_seq_len * batch, dtype="int64")
    tokens = torch.from_numpy(tokens).long().reshape(batch, max_seq_len).cuda()

    model_dict = {}
    torch_decoder = TransformerDecoder(args, dictionary, embedding, True).cuda().eval()

    # eet 需要传入数据类型、最大的batch_size,以及提示词长度，该长度可根据具体业务判断最长会到多长去设定。
    eet_config = {"data_type": torch.float32, "max_batch": batch, "full_seq_len": max_seq_len}
    eet_model = EETTransformerDecoder.from_torch(torch_decoder=torch_decoder, dictionary=dictionary, args=args, config=eet_config, no_encoder_attn=True)

    torch.cuda.synchronize()
    t1 = time.perf_counter()

    # 推理，多次循环做耗时比较
    for i in range(loop):
        '''
        first_pass 用于判断生成任务时是否是第一步，也就是是否是在做提示词的推理。true代表在做提示词的推理，false代表在做生成推理
        由于eet不会返回past_key_value，前一步的信息全部在内部做了保存，所以没法通过past_key_value做判断，故增加此参数。
        reorder_state 指的是输入的文本可以动态调整位置，支持提前结束。
        譬如四个句子组成一组batch_size=4的输入，在推理过程中，某个句子提前结束，或者句子的顺序发生了变化，可通过reorder_stata去做调整，从而节约资源。
        '''
        first_pass = True
        reorder_state = None
        for step in range(context_len - 1, max_seq_len):
            if first_pass:
                input_ids_eet = tokens[:, :step + 1].contiguous().cuda().long()
            else:
                input_ids_eet = tokens[:, step:step + 1].contiguous().cuda().long()
            res_eet = eet_model(input_ids_eet, reorder_state=reorder_state, first_pass=first_pass)
            first_pass = False

    torch.cuda.synchronize()
    t2 = time.perf_counter()
    print('Time for EET : ', t2 - t1)

    torch.cuda.synchronize()
    t3 = time.perf_counter()

    for i in range(loop):
        incremental_state = {}
        for step in range(0, max_seq_len):
            res_torch, incremental_state = torch_decoder(tokens[:, :step+1], incremental_state=incremental_state)

    torch.cuda.synchronize()
    t4 = time.perf_counter()

    print('Time for Fairseq : ', t4 - t3)
    print('SpeedUp is : ', (t4 - t3)/(t2 - t1))


if __name__ == '__main__':
    main()
