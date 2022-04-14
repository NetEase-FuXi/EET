import torch
import time
import numpy as np
from torch import nn
from fairseq.data.dictionary import Dictionary
from eet.fairseq.transformer import EETTransformerDecoder

using_eet = True
using_half = True

prompt_seq_len = 512

# eet supports a maximum seq_len of 4096 
max_seq_len = 1024
batch = 4

def main():
    # Model file path, it should be noted that the model file name must be checkpoint_best.pt,If not, please change it to the same, or modify the source code python/eet/fairseq/transformer.py line 451
    model_id_or_path = '../resource/'
    torch.set_grad_enabled(False)
    tokens = np.random.randint(3,13672,max_seq_len * batch,dtype="int64")
    tokens = torch.from_numpy(tokens).long().reshape(batch, max_seq_len).cuda()
    data_type = torch.float32
    if using_half:
        data_type = torch.float16
    eet_model = EETTransformerDecoder.from_pretrained(model_id_or_path = model_id_or_path,max_batch = batch, full_seq_len = prompt_seq_len,data_type = data_type,no_encoder_attn = True)


    total_time_eet = 0
    '''
    first_pass 用于判断生成任务时是否是第一步，也就是是否是在做提示词的推理。true代表在做提示词的推理，false代表在做生成推理
    由于eet不会返回past_key_value，前一步的信息全部在内部做了保存，所以没法通过past_key_value做判断，故增加此参数。
    reorder_state 指的是输入的文本可以动态调整位置，支持提前结束。
    譬如四个句子组成一组batch_size=4的输入，在推理过程中，某个句子提前结束，或者句子的顺序发生了变化，可通过reorder_stata去做调整，从而节约资源。
    '''
    first_pass = True
    reorder_state = None
    for step in range(prompt_seq_len-1, max_seq_len):
        torch.cuda.synchronize()
        t1 = time.perf_counter()
        if first_pass:
            input_ids_eet = torch.clone(tokens[:, :step + 1].contiguous()).cuda().long()
        else:
            input_ids_eet = torch.clone(tokens[:, step:step + 1].contiguous()).cuda().long()
        res_eet = eet_model(input_ids_eet, reorder_state = reorder_state,first_pass = first_pass)
        torch.cuda.synchronize()
        t2 = time.perf_counter()
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
