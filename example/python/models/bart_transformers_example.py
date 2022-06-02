import torch
import numpy as np
from torch.nn.parameter import Parameter
from eet.transformers.modeling_bart import EETBartModel
from transformers import BartModel
from PIL import Image
import requests
import time


def main():
    torch.set_printoptions(precision=3, sci_mode=False)
    torch.set_grad_enabled(False)
    # 输入数据构造，实际业务输入应该是tokens
    # inputs = np.random.randint(1000, 9000, seq_len * batch_size, dtype='int64')
    inputs = np.ones((batch_size, seq_len), dtype='int64')
    input_ids = torch.from_numpy(inputs).long().reshape(batch_size, seq_len).cuda()
    
    data_type = torch.float16 if using_half else torch.float32

    # load model
    ts_model = BartModel.from_pretrained('facebook/bart-base').cuda()
    ts_model = ts_model.half() if using_half else ts_model
    eet_model = EETBartModel.from_pretrained('facebook/bart-base', batch_size, seq_len, data_type=data_type)
    attention_mask = None

    '''
    first_pass 用于判断生成任务时是否是第一步，也就是是否是在做提示词的推理。true代表在做提示词的推理，false代表在做生成推理
    由于eet不会返回past_key_value，前一步的信息全部在内部做了保存，所以没法通过past_key_value做判断，故增加此参数。
    '''
    for i in range(loop):
        res_eet = eet_model(input_ids=input_ids, attention_mask=attention_mask, first_pass=True)
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

def test_perf(batch_size=4, seq_len=8, using_half=True, loop=100):
    torch.set_printoptions(precision=3, sci_mode=False)
    torch.set_grad_enabled(False)
    # 输入数据构造，实际业务输入应该是tokens
    inputs = np.random.randint(1000, 9000, seq_len * batch_size, dtype='int64')
    # inputs = np.ones((batch_size, seq_len), dtype='int64')
    input_ids = torch.from_numpy(inputs).long().reshape(batch_size, seq_len).cuda()
    data_type = torch.float16 if using_half else torch.float32

    # load model
    ts_model = BartModel.from_pretrained('facebook/bart-base').cuda()
    ts_model = ts_model.half() if using_half else ts_model
    eet_model = EETBartModel.from_pretrained('facebook/bart-base', batch_size, seq_len, data_type=data_type)
    attention_mask = None

    '''
    first_pass 用于判断生成任务时是否是第一步，也就是是否是在做提示词的推理。true代表在做提示词的推理，false代表在做生成推理
    由于eet不会返回past_key_value，前一步的信息全部在内部做了保存，所以没法通过past_key_value做判断，故增加此参数。
    '''

    for i in range(loop):
        res_ts = ts_model(input_ids=input_ids, attention_mask=attention_mask)
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
    fp = 'fp16' if using_half else 'fp32'
    print('*****************************')
    print(batch_size, '\t', seq_len, '\t', fp)

def test(batch_size=4, seq_len=8, max_seq_len=1024, using_half=True, using_eet=False, using_ts=True, loop=100):
    torch.set_printoptions(precision=3, sci_mode=False)
    torch.set_grad_enabled(False)
    # 输入数据构造，实际业务输入应该是tokens
    # inputs = np.random.randint(1000, 9000, seq_len * batch_size, dtype='int64')
    input_full = np.ones((batch_size, seq_len), dtype='int64')
    input_inc = np.ones((batch_size, 1), dtype='int64')
    # input_ids = torch.from_numpy(inputs).long().reshape(batch_size, seq_len).cuda()
    input_full_decoder = torch.from_numpy(input_full).long().reshape(batch_size, seq_len).cuda()
    input_inc_decoder = torch.from_numpy(input_inc).long().reshape(batch_size, 1).cuda()

    data_type = torch.float16 if using_half else torch.float32
    attention_mask = None
    
    if using_ts:
        ts_model = BartModel.from_pretrained('facebook/bart-base').cuda()
        ts_model = ts_model.half() if using_half else ts_model   
        torch.cuda.synchronize()
        t3 = time.perf_counter()
        for i in range(loop):
            input_ids = input_full_decoder
            past_key_values = None
            encoder_outputs = None
            for j in range(max_seq_len - seq_len):
                with torch.no_grad():
                    res_ts = ts_model(input_ids=input_ids, past_key_values=past_key_values, attention_mask=attention_mask, encoder_outputs=encoder_outputs)
                past_key_values = res_ts.past_key_values
                print(j, ' res ts: ', res_ts.last_hidden_state, ' shape: ', res_ts.last_hidden_state.size())
                input_ids = input_inc_decoder
                encoder_outputs = (res_ts.encoder_last_hidden_state, )
        torch.cuda.synchronize()
        t4 = time.perf_counter()
        time_ts = t4 - t3

    if using_eet:
    # load model
        eet_model = EETBartModel.from_pretrained('facebook/bart-base', batch_size, seq_len, data_type=data_type)

        '''
        first_pass 用于判断生成任务时是否是第一步，也就是是否是在做提示词的推理。true代表在做提示词的推理，false代表在做生成推理
        由于eet不会返回past_key_value，前一步的信息全部在内部做了保存，所以没法通过past_key_value做判断，故增加此参数。
        '''
        for i in range(loop):
            res_eet = eet_model(input_ids=input_full_decoder, attention_mask=attention_mask, first_pass=True)
        
        torch.cuda.synchronize()
        t1 = time.perf_counter()
        for i in range(loop):
            input_ids = input_full_decoder
            self_past_key_values_length = 0
            first_pass = True
            for j in range(max_seq_len - seq_len):
                res_eet = eet_model(input_ids=input_ids, attention_mask=attention_mask, first_pass=first_pass, self_past_key_values_length=self_past_key_values_length)
                self_past_key_values_length += input_ids.shape[1]
                print(j, ' res eet: ', res_eet, ' shape: ', res_eet.shape)
                if first_pass:
                    first_pass = False
                input_ids = input_inc_decoder
        torch.cuda.synchronize()
        t2 = time.perf_counter()
        time_eet = t2 - t1

    if using_eet and using_ts:
        print('Time for eet: ', time_eet)
        print('Time for Transformers: ', time_ts)
        print('SpeedUp is ', time_ts / time_eet)
    elif using_eet:
        print('Time for eet: ', time_eet)
    elif using_ts:
        print('Time for Transformers: ', time_ts)
    fp = 'fp16' if using_half else 'fp32'
    print('*****************************')
    print(batch_size, '\t', seq_len, '\t', fp)


if __name__ == '__main__':
    test(batch_size=4, seq_len=10, max_seq_len=12, using_half=True,
         using_eet=True, using_ts=True, loop=1)
