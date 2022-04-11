import torch
import numpy as np
from eet import EETGPT2Model
from transformers import GPT2Model
import time
using_half = False
seq_len = 511
batch = 5

def main():
    input = np.random.randint(1000,9000,seq_len * batch,dtype="int64")
    inputs = np.random.randint(1000,9000,1 * batch,dtype="int64")
    # prompt context
    input_full_decoder = torch.from_numpy(input).long().reshape(batch, seq_len).cuda()

    # prediction 
    input_inc_decoder = torch.from_numpy(inputs).long().reshape(batch, 1).cuda()

    data_type = torch.float32
    if using_half:
        data_type = torch.float16

    # load model
    eet_model = EETGPT2Model.from_pretrained('gpt2',max_batch = batch, full_seq_len = seq_len,data_type = data_type)
    torch_model = GPT2Model.from_pretrained('gpt2').cuda()
    if using_half:
        torch_model =torch_model.half()

    torch.cuda.synchronize()
    t1 = time.perf_counter()
    for j in range(10):
        input_ids = input_full_decoder
        attention_mask = None
        first_pass = True
        for i in range(512):
            res_eet = eet_model(input_ids,first_pass= first_pass,attention_mask = attention_mask)
            if first_pass:
                first_pass = False
            input_ids = input_inc_decoder
    torch.cuda.synchronize()
    t2 = time.perf_counter()
    print('Time for EET : ', t2 - t1)


    torch.cuda.synchronize()
    t3 = time.perf_counter()
    for j in range(10):
        input_ids = input_full_decoder
        past_key_values = None
        for i in range(512):
            with torch.no_grad():
                res_torch = torch_model(input_ids,past_key_values = past_key_values,attention_mask = attention_mask)
            past_key_values = res_torch.past_key_values
            input_ids = input_inc_decoder
    torch.cuda.synchronize()
    t4 = time.perf_counter()

    print('Time for torch : ', t4 - t3)
    print('SpeedUp is : ', (t4 - t3)/(t2- t1))

if __name__ == '__main__':
    main()
