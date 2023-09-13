import torch
import numpy as np
from eet import EETMegatronBertForMaskedLM
from transformers import MegatronBertForMaskedLM,MegatronBertModel
import time

import random

using_half = True
seq_len = 256
batch = 64
loop = 100

def main():
    torch.set_grad_enabled(False)
    torch.manual_seed(111)
    random.seed(111)
    np.random.seed(111)
    # 输入数据构造，实际业务输入应该是tokens
    input = np.random.randint(1000, 9000, seq_len * batch, dtype="int64")
    input_ids = torch.from_numpy(input).long().reshape(batch, seq_len).cuda()

    data_type = torch.float32
    if using_half:
        data_type = torch.float16
    
    # load model,eet 需要传入最大batch_size和数据类型
    eet_model = EETMegatronBertForMaskedLM.from_pretrained('/root/L36_NLP/MegatronBert/MegatronBertForMaskedLM', max_batch=batch, data_type=data_type)
    ts_model = MegatronBertForMaskedLM.from_pretrained('/root/L36_NLP/MegatronBert/MegatronBertForMaskedLM').cuda()
    if using_half:
        ts_model = ts_model.half()
    attention_mask = None

    # warmup
    for i in range(10):
        res_eet = eet_model(input_ids, attention_mask=attention_mask)

    # inference
    torch.cuda.synchronize()
    t1 = time.perf_counter()
    for i in range(loop):
        res_eet = eet_model(input_ids, attention_mask=attention_mask)
        # print(res_eet)
    torch.cuda.synchronize()

    t2 = time.perf_counter()
    time_eet = t2 - t1
    print('Time for EET : ', time_eet)
    torch.cuda.synchronize()

    t3 = time.perf_counter()
    with torch.no_grad():
        for i in range(loop):
            res_ts = ts_model(input_ids, attention_mask)
            # print(res_ts)

    torch.cuda.synchronize()

    t4 = time.perf_counter()
    time_ts = t4 -t3

    
    print('Time for Transformers: ', time_ts)
    print('SpeedUp is ', time_ts / time_eet)

if __name__ == '__main__':
    main()
