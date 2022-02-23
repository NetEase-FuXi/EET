import torch
import numpy as np
from eet.transformers.modeling_roberta import EETRobertaModel
from transformers import RobertaModel
import time

using_half = True
seq_len = 128
batch = 4
loop = 100

def main():
    torch.set_grad_enabled(False)

    print("bs: ", batch, " seq_len: ", seq_len)
    input = np.random.randint(1000, 9000, seq_len * batch, dtype="int64")
    input_ids = torch.from_numpy(input).long().reshape(batch, seq_len).cuda()

    data_type = torch.float32
    ts_model = RobertaModel.from_pretrained('roberta-base').cuda()
    if using_half:
        ts_model = ts_model.half()
        data_type = torch.float16
    eet_model = EETRobertaModel.from_pretrained('roberta-base',max_batch = batch,data_type = data_type)
  
    attention_mask = None
    t1 = time.perf_counter()
    for i in range(loop):
        res_eet = eet_model(input_ids, attention_mask=attention_mask)
    t2 = time.perf_counter()
    time_eet = t2 - t1
    print('Time for EET : ', time_eet)

    t3 = time.perf_counter()
    for i in range(loop):
        res_ts = ts_model(input_ids, attention_mask)
    t4= time.perf_counter()
    time_ts = t4 - t3
    
    print('Time for Transformers: ', time_ts)
    print('SpeedUp is ', time_ts / time_eet)

if __name__ == '__main__':
    main()
