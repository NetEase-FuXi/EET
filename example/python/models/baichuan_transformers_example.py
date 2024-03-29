import os
import time
import psutil
import random
import torch
import numpy as np
from torch.nn.parameter import Parameter
from transformers import AutoTokenizer, AutoConfig, AutoModelForCausalLM
from eet.transformers.modeling_baichuan import convert_baichuan_weights

model_path = "/root/download/baichuan2-7b-base/"
int8_ckpt_path = "/root/download/baichuan2-7b-base/eet_baichuan_int8.pt"
fp16_ckpt_path = "/root/download/baichuan2-7b-base/eet_baichuan_fp16.pt"

def set_random_seed(seed):
    random.seed(seed)
    torch.manual_seed(seed)
    torch.backends.cudnn.deterministic = True
    torch.cuda.manual_seed_all(seed)
    np.random.seed(seed)


def save_dict():
    free_in_GB = int(torch.cuda.mem_get_info()[0]/1024**3)
    max_memory = f'{int(torch.cuda.mem_get_info()[0]/1024**3)-2}GB'
    n_gpus = torch.cuda.device_count()
    max_memory = {i: max_memory for i in range(n_gpus)}

    # ts_model = AutoModelForCausalLM.from_pretrained(
    #     model_path,
    #     device_map=0,
    #     load_in_8bit=True,
    #     max_memory=max_memory
    # )
    config = AutoConfig.from_pretrained(model_path, trust_remote_code=True)
    # config.num_hidden_layers = 1
    ts_model = AutoModelForCausalLM.from_pretrained(model_path, config=config, trust_remote_code=True, torch_dtype=torch.float16)
    ts_model = ts_model.half()
    num_params = sum(p.numel() for p in ts_model.parameters())
    print("num_params: ", num_params)

    eet_baichuan_dict = convert_baichuan_weights(ts_model.state_dict(), data_type=torch.int8)
    print("*******************after quant*****************")
    for k, v in eet_baichuan_dict.items():
        print(k, v.shape)
    torch.save(eet_baichuan_dict, int8_ckpt_path)

def test_torch_inference(batch_size=1, prompt_seq_len=1024, max_new_tokens=50, loop=10):
    torch.set_printoptions(precision=6, sci_mode=False)
    torch.set_grad_enabled(False)
    tokenizer = AutoTokenizer.from_pretrained(model_path, trust_remote_code=True)
    config = AutoConfig.from_pretrained(model_path, trust_remote_code=True)
    # transformer
    ts_lm_model = AutoModelForCausalLM.from_pretrained(model_path, config=config, trust_remote_code=True)
    ts_model = ts_lm_model.to(torch.float16).model.cuda()

    # dummy input
    input_full = np.random.randint(2000, 3000, prompt_seq_len * batch_size, dtype='int64')
    input_inc = np.random.randint(2000, 3000, 1 * batch_size, dtype='int64')
    input_full_decoder = torch.from_numpy(input_full).long().reshape(batch_size, prompt_seq_len).cuda()
    input_inc_decoder = torch.from_numpy(input_inc).long().reshape(batch_size, 1).cuda()

    # # warm up
    # for i in range(2):
    #     res_ts = ts_model(input_ids=input_full_decoder, past_key_values=None, attention_mask=None)

    # profile
    torch.cuda.synchronize()
    t1 = time.perf_counter()
    for i in range(loop):
        input_ids = input_full_decoder
        res_ts = ts_model(input_ids=input_ids, past_key_values=None, attention_mask=None)
        past_key_values = res_ts.past_key_values
        for j in range(max_new_tokens):
            res_ts = ts_model(input_ids=input_ids, past_key_values=past_key_values, attention_mask=None)
            # print("step: ", j, " eet output: ", res_eet.reshape(-1)[:12800:640])
            input_ids = input_inc_decoder
    torch.cuda.synchronize()
    t2 = time.perf_counter()
    time_eet = t2 - t1
    print("batch_size: {}, prompt_length: {}, max_new_tokens: {}, loop: {}, lantency: {:.4f} s".format(batch_size, prompt_seq_len, max_new_tokens, loop, time_eet / loop))
    print('Time for ts: ', time_eet / loop)
    max_memory_allocated = torch.cuda.max_memory_allocated() / 1e9
    print('当前进程号: {}, 内存使用：{:.4f} GB'.format(os.getpid(), psutil.Process(os.getpid()).memory_info().rss / 1024 / 1024 / 1024))
    print("max GPU memory allocated: {:.4f} GB".format(max_memory_allocated))

def test_eet_inference(batch_size=1, prompt_seq_len=332, max_new_tokens=50, loop=10):
    from eet import EETBaichuanModel, EETBaichuanForCausalLM

    torch.set_printoptions(precision=6, sci_mode=False)
    torch.set_grad_enabled(False)
    set_random_seed(1)
    tokenizer = AutoTokenizer.from_pretrained(model_path, trust_remote_code=True, use_fast=False)
    config = AutoConfig.from_pretrained(model_path, trust_remote_code=True)
    
    attention_mask = None

    eet_model = EETBaichuanForCausalLM.from_pretrained(int8_ckpt_path, config, max_batch=batch_size, max_prompt_seq_len=prompt_seq_len,
                                                       max_full_seq_len=prompt_seq_len+max_new_tokens+1, data_type=torch.int8)

    # dummy input
    input_full = np.random.randint(2000, 3000, prompt_seq_len * batch_size, dtype='int64')
    input_inc = np.random.randint(2000, 3000, 1 * batch_size, dtype='int64')
    
    input_full_decoder = torch.from_numpy(input_full).long().reshape(batch_size, prompt_seq_len).cuda()
    input_inc_decoder = torch.from_numpy(input_inc).long().reshape(batch_size, 1).cuda()

    # ts_model = AutoModelForCausalLM.from_pretrained(model_path, config=config, trust_remote_code=True, torch_dtype=torch.float16)
    # ts_model.half().to("cuda:0")
    # ts_output = ts_model(input_ids=input_full_decoder, attention_mask=attention_mask)
    # print("ts output: ", ts_output.logits.shape, ts_output.logits.reshape(-1)[-12800::640])
    # eet_output = eet_model(input_ids=input_full_decoder, first_pass=True, attention_mask=attention_mask)
    # print("eet output: ", eet_output.logits.shape, eet_output.logits.reshape(-1)[-12800::640])

    # warm up
    # for i in range(2):
    #     res_eet = eet_model(input_ids=input_full_decoder, first_pass=True, attention_mask=attention_mask)
    #     # print(res_eet)
    # profile
    torch.cuda.synchronize()
    t1 = time.perf_counter()
    for i in range(loop):
        input_ids = input_full_decoder
        first_pass = True
        for j in range(max_new_tokens):
            with torch.no_grad():
                res_eet = eet_model(input_ids=input_ids, first_pass=first_pass, attention_mask=attention_mask)
            # print("step: ", j, " eet output: ", res_eet.reshape(-1)[:12800:640])
            if first_pass:
                first_pass = False
            input_ids = input_inc_decoder
    torch.cuda.synchronize()
    t2 = time.perf_counter()
    time_eet = t2 - t1
    print("batch_size: {}, prompt_length: {}, max_new_tokens: {}, loop: {}, lantency: {:.4f} s".format(batch_size, prompt_seq_len, max_new_tokens, loop, time_eet / loop))
    print('Time for eet: ', time_eet / loop)
    max_memory_allocated = torch.cuda.max_memory_allocated() / 1e9
    print('当前进程号: {}, 内存使用：{:.4f} GB'.format(os.getpid(), psutil.Process(os.getpid()).memory_info().rss / 1024 / 1024 / 1024))
    print("max GPU memory allocated: {:.4f} GB".format(max_memory_allocated))


def test_eet_generate_int8(batch_size=1, prompt_seq_len=1024, max_new_tokens=50):
    from eet import EETBaichuanModel, EETBaichuanForCausalLM

    torch.set_printoptions(precision=6, sci_mode=False)
    torch.set_grad_enabled(False)
    set_random_seed(1)
    tokenizer = AutoTokenizer.from_pretrained(model_path, trust_remote_code=True, use_fast=False)
    config = AutoConfig.from_pretrained(model_path, trust_remote_code=True)

    attention_mask = None
    eet_model = EETBaichuanForCausalLM.from_pretrained(int8_ckpt_path, config, max_batch=batch_size, max_prompt_seq_len=prompt_seq_len,
                                                       max_full_seq_len=prompt_seq_len+max_new_tokens+1, data_type=torch.int8)

    text = '解释一下“温故而知新”'
    kwargs = {
        "max_new_tokens": int(max_new_tokens),
        "min_new_tokens": int(10),
        "do_sample": bool(False),
        # "temperature": float(0.75),
        "top_k": int(50),
        # "top_p": float(0.7),
        "use_cache": bool(True),
    }
    inputs = tokenizer(str(text), return_tensors='pt')
    kwargs["inputs"] = inputs.input_ids.to('cuda')
    # input_ids = torch.randint(1000, 8000, (batch_size, prompt_seq_len), dtype=torch.long, device='cuda')
    # kwargs["inputs"] = input_ids.to('cuda')
    # generate
    generate_ids = eet_model.generate(**kwargs)
    outputs_str = tokenizer.batch_decode(
        generate_ids, 
        skip_special_tokens=True, 
        clean_up_tokenization_spaces=False)
    print("outputs_str", outputs_str)
    max_memory_allocated = torch.cuda.max_memory_allocated() / 1e9
    print("batch_size: {}, prompt_length: {}, max_new_tokens: {}".format(batch_size, prompt_seq_len, max_new_tokens))
    print('当前进程号: {}, 内存使用：{:.4f} GB'.format(os.getpid(), psutil.Process(os.getpid()).memory_info().rss / 1024 / 1024 / 1024))
    print("max GPU memory allocated: {:.4f} GB".format(max_memory_allocated))


def test_eet_generate_fp16(batch_size=1, prompt_seq_len=1024, max_new_tokens=50):
    from eet import EETBaichuanModel, EETBaichuanForCausalLM

    torch.set_printoptions(precision=6, sci_mode=False)
    torch.set_grad_enabled(False)
    set_random_seed(1)
    tokenizer = AutoTokenizer.from_pretrained(model_path, trust_remote_code=True, use_fast=False)
    config = AutoConfig.from_pretrained(model_path, trust_remote_code=True)

    attention_mask = None
    eet_model = EETBaichuanForCausalLM.from_pretrained(model_path, config, max_batch=batch_size, max_prompt_seq_len=prompt_seq_len,
                                                       max_full_seq_len=prompt_seq_len+max_new_tokens+1, data_type=torch.float16)

    text = '解释一下“温故而知新”'
    kwargs = {
        "max_new_tokens": int(max_new_tokens),
        "min_new_tokens": int(10),
        "do_sample": bool(False),
        # "temperature": float(0.75),
        "top_k": int(50),
        # "top_p": float(0.7),
        "use_cache": bool(True),
    }
    inputs = tokenizer(str(text), return_tensors='pt')
    kwargs["inputs"] = inputs.input_ids.to('cuda')
    # input_ids = torch.randint(1000, 8000, (batch_size, prompt_seq_len), dtype=torch.long, device='cuda')
    # kwargs["inputs"] = input_ids.to('cuda')
    # generate
    generate_ids = eet_model.generate(**kwargs)
    outputs_str = tokenizer.batch_decode(
        generate_ids, 
        skip_special_tokens=True, 
        clean_up_tokenization_spaces=False)
    print("outputs_str", outputs_str)
    max_memory_allocated = torch.cuda.max_memory_allocated() / 1e9
    print("batch_size: {}, prompt_length: {}, max_new_tokens: {}".format(batch_size, prompt_seq_len, max_new_tokens))
    print('当前进程号: {}, 内存使用：{:.4f} GB'.format(os.getpid(), psutil.Process(os.getpid()).memory_info().rss / 1024 / 1024 / 1024))
    print("max GPU memory allocated: {:.4f} GB".format(max_memory_allocated))


if __name__ == "__main__":
    # save_dict()
    # test_torch_inference(batch_size=4, prompt_seq_len=1024, max_new_tokens=50, loop=10)
    # test_eet_inference(batch_size=5, prompt_seq_len=1300, max_new_tokens=50, loop=10)
    # test_eet_generate_int8(batch_size=1, prompt_seq_len=1024, max_new_tokens=50)
    test_eet_generate_fp16(batch_size=1, prompt_seq_len=1024, max_new_tokens=50)
