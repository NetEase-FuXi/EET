import torch
import numpy as np
from torch.nn.parameter import Parameter
from transformers import T5Model, T5Tokenizer
from eet.transformers.modeling_t5 import EETT5Model
from PIL import Image
import requests
import time

using_half = True
batch_size = 4
seq_len = 512
max_seq_len = 1024
loop = 10
device = "cuda" if torch.cuda.is_available() else "cpu"


def main():
    torch.set_printoptions(precision=3, sci_mode=False)
    torch.set_grad_enabled(False)
    # 输入数据构造，实际业务输入应该是tokens
    # tokenizer = T5Tokenizer.from_pretrained("t5-small")
    # inputs = tokenizer("Hello, my dog is cute", return_tensors="pt")
    # fake data
    input_full = np.random.randint(1000, 9000, seq_len * batch_size, dtype='int64')
    input_inc = np.random.randint(1000, 9000, 1 * batch_size, dtype='int64')

    input_full_decoder = torch.from_numpy(input_full).long().reshape(batch_size, seq_len).cuda()
    input_inc_decoder = torch.from_numpy(input_inc).long().reshape(batch_size, 1).cuda()

    data_type = torch.float16 if using_half else torch.float32
    attention_mask = None

    # compute per sample length EET增量推理需要输入encoder out的真实句长
    encoder_seq_len = torch.tensor([seq_len] * batch_size).int().cuda()
    if attention_mask is not None:
        pre_padding_len = torch.sum(1 - attention_mask, 1).int().cuda()
        encoder_seq_len = encoder_seq_len - pre_padding_len

    # load transformers model
    ts_model = T5Model.from_pretrained('t5-small').cuda()
    ts_model = ts_model.half() if using_half else ts_model
    # warm up
    for i in range(loop):
        res_ts = ts_model(input_ids=input_full_decoder, decoder_input_ids=input_full_decoder)

    torch.cuda.synchronize()
    t3 = time.perf_counter()
    for i in range(loop):
        input_ids = input_full_decoder
        past_key_values = None
        encoder_outputs = None
        for j in range(max_seq_len - seq_len):
            with torch.no_grad():
                res_ts = ts_model(input_ids=input_ids, decoder_input_ids=input_ids, past_key_values=past_key_values, attention_mask=attention_mask, encoder_outputs=encoder_outputs)
            past_key_values = res_ts.past_key_values
            input_ids = input_inc_decoder
            encoder_outputs = (res_ts.encoder_last_hidden_state, )
    torch.cuda.synchronize()
    t4 = time.perf_counter()
    time_ts = t4 - t3

    # load eet model
    eet_model = EETT5Model.from_pretrained('t5-small', batch_size, data_type=data_type)
    '''
    first_pass 用于判断生成任务时是否是第一步，也就是是否是在做提示词的推理。true代表在做提示词的推理，false代表在做生成推理
    由于eet不会返回past_key_value，前一步的信息全部在内部做了保存，所以没法通过past_key_value做判断，故增加此参数。
    '''
    torch.cuda.synchronize()
    t1 = time.perf_counter()
    for i in range(loop):
        input_ids = input_full_decoder
        self_past_key_values_length = 0
        first_pass = True
        for j in range(max_seq_len - seq_len):
            res_eet = eet_model(input_ids=input_ids, decoder_input_ids=input_ids, attention_mask=attention_mask, encoder_seq_length=encoder_seq_len, first_pass=first_pass, self_past_key_values_length=self_past_key_values_length)
            self_past_key_values_length += input_ids.shape[1]
            if first_pass:
                first_pass = False
            input_ids = input_inc_decoder
    torch.cuda.synchronize()
    t2 = time.perf_counter()
    time_eet = t2 - t1

    print('Time for eet: ', time_eet)
    print('Time for Transformers: ', time_ts)
    print('SpeedUp is ', time_ts / time_eet)


if __name__ == '__main__':
    main()
