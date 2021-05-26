import torch
import numpy as np
from eet.transformers.modeling_bert import EETBertModel
using_half = False

seq_len = 500
batch = 3

def main():
    torch.set_grad_enabled(False)

    input = np.random.randint(1000,9000,seq_len * batch,dtype="int64")
    input_full = torch.from_numpy(input).long().reshape(batch, seq_len).cuda()

    data_type = torch.float32
    if using_half:
        data_type = torch.float16
    eet_model = EETBertModel.from_pretrained('bert-base-uncased',max_batch = batch,data_type = data_type)
  
    input_ids = input_full
    attention_mask = None
    for i in range(50):
        print('i--:',i)
        res_eet = eet_model(input_ids, attention_mask=attention_mask)

if __name__ == '__main__':
    main()
