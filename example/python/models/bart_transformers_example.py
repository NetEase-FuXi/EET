import torch
import numpy as np
from torch.nn.parameter import Parameter
from eet.transformers.modeling_bart_fp16bug import EETBartModel
from transformers import BartModel
from PIL import Image
import requests
import time

using_half = False
batch_size = 1
seq_len = 8
loop = 1
device = "cuda" if torch.cuda.is_available() else "cpu"


def main():
    torch.set_printoptions(precision=3, sci_mode=False)
    torch.set_grad_enabled(False)
    # 输入数据构造，实际业务输入应该是tokens
    inputs = np.random.randint(1000, 9000, seq_len * batch_size, dtype='int64')
    input_ids = torch.from_numpy(inputs).long().contiguous().view(batch_size, seq_len).cuda()
    
    data_type = torch.float16 if using_half else torch.float32

    # load model
    ts_model = BartModel.from_pretrained('facebook/bart-base').cuda()
    eet_model = EETBartModel.from_pretrained('facebook/bart-base', batch_size, seq_len, data_type=data_type)
    if using_half:
        ts_model = ts_model.half()
    attention_mask = None

    '''
    first_pass 用于判断生成任务时是否是第一步，也就是是否是在做提示词的推理。true代表在做提示词的推理，false代表在做生成推理
    由于eet不会返回past_key_value，前一步的信息全部在内部做了保存，所以没法通过past_key_value做判断，故增加此参数。
    '''
    # for i in range(loop):
    #     res_eet = eet_model(input_ids=input_ids, attention_mask=attention_mask, first_pass=True)
    # print('res eet: ', res_eet)
    
    torch.cuda.synchronize()
    t1 = time.perf_counter()
    for i in range(loop):
        res_eet = eet_model(input_ids=input_ids, attention_mask=attention_mask, first_pass=True)
    torch.cuda.synchronize()
    t2 = time.perf_counter()
    time_eet = t2 - t1
    
    torch.cuda.synchronize()
    t3 = time.perf_counter()
    for i in range(loop):
        with torch.no_grad():
            res_ts = ts_model(input_ids=input_ids, attention_mask=attention_mask)
    torch.cuda.synchronize()
    t4 = time.perf_counter()
    time_ts = t4 - t3

    print('res ts: ', res_ts.last_hidden_state)
    print('res eet: ', res_eet)
    print('Time for eet: ', time_eet)
    print('Time for Transformers: ', time_ts)
    print('SpeedUp is ', time_ts / time_eet)


if __name__ == '__main__':
    main()