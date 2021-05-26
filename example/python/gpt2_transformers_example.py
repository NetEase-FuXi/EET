import torch
import numpy as np
from eet.transformers.modeling_gpt2 import EETGPT2Model
using_half = False
seq_len = 128
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

    # load pytorch model
    eet_model = EETGPT2Model.from_pretrained('gpt2',max_batch = batch, full_seq_len = seq_len,data_type = data_type)
    input_ids = input_full_decoder

    first_pass = True
    for i in range(100):
        print('i--:',i)
        res_eet = eet_model(input_ids,first_pass= first_pass)
        if first_pass:
            first_pass = False
        input_ids = input_inc_decoder

if __name__ == '__main__':
    main()
