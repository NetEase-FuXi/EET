import os
import time
import psutil
import random
import torch
import numpy as np
from torch.nn.parameter import Parameter
import torch.nn as nn
from eet import EETLlamaModel, EETLlamaForCausalLM
from transformers import AutoTokenizer, LlamaModel, LlamaForSequenceClassification, LlamaForCausalLM, LlamaTokenizer,AutoConfig,AutoModelForCausalLM

# model_dir = "decapoda-research/llama-7b-hf"
model_dir = "/root/download/llama-13b"

def set_random_seed(seed):
    random.seed(seed)
    torch.manual_seed(seed)
    torch.backends.cudnn.deterministic = True
    torch.cuda.manual_seed_all(seed)
    np.random.seed(seed)

def test():
    print("*************")
    tokenizer = LlamaTokenizer.from_pretrained(model_dir)
    config = AutoConfig.from_pretrained("/root/download/llama-7b")

    prompt = "你好，介绍一下你自己?"
    inputs = tokenizer(prompt, return_tensors="pt")
    input_ids = inputs["input_ids"].cuda()
    model = LlamaForCausalLM.from_pretrained(model_dir, config=config, torch_dtype=torch.float16)

    for k, v in model.state_dict().items():
        print(k, v.shape)
    # print(model.config)
    # Generate
    model.half().cuda()
    generate_ids = model.generate(input_ids, max_length=128)
    # print logits
    # print("logits: ", model(input_ids).logits)
    # outputs = tokenizer.batch_decode(generate_ids, skip_special_tokens=True, clean_up_tokenization_spaces=False)[0]
    # print("outputs: ", outputs)

    print("***********************************")
    max_memory_allocated = torch.cuda.max_memory_allocated() / 1e9
    print('当前进程号: {}, 内存使用：{:.4f} GB'.format(os.getpid(), psutil.Process(os.getpid()).memory_info().rss / 1024 / 1024 / 1024))
    print("max GPU memory allocated: {:.4f} GB".format(max_memory_allocated))

def test_eet():
    print("*************")
    model_dir = "/root/download/llama-13b"
    tokenizer = LlamaTokenizer.from_pretrained(model_dir)
    config = AutoConfig.from_pretrained(model_dir)
    prompt = "你好，介绍一下你自己?"
    inputs = tokenizer(prompt, return_tensors="pt")
    input_ids = inputs["input_ids"].cuda()
    model_dict = {}
    llama_dict = {}
    lm_head_dict = {}
    
    model_dict = LlamaForCausalLM.from_pretrained(model_dir).state_dict()

    for k, v in model_dict.items():
        if 'lm_head' in k:
            k = k[k.find('weight'):]
            lm_head_dict[k] = v
        else:
            llama_dict[k] = v

    llamamodel = EETLlamaModel.from_pretrained(config, llama_dict, max_batch=1, max_prompt_seq_len=1024,
                                                max_full_seq_len=2048, data_type=torch.float16, model_attr='model', is_int8=False)

    # torch_model.to(data_type)
    import torch.nn as nn
    lm_head = nn.Linear(config.hidden_size, config.vocab_size, bias=False)
    lm_head.load_state_dict(lm_head_dict)
    lm_head = lm_head.half().cuda()
    eet_model = EETLlamaForCausalLM(config, llamamodel, lm_head)
    # Generate
    generate_ids = eet_model.generate(input_ids, max_length=128)
    outputs = tokenizer.batch_decode(generate_ids, skip_special_tokens=True, clean_up_tokenization_spaces=False)
    print("outputs: ", outputs)

    print("***********************************")
    max_memory_allocated = torch.cuda.max_memory_allocated() / 1e9
    print('当前进程号: {}, 内存使用：{:.4f} GB'.format(os.getpid(), psutil.Process(os.getpid()).memory_info().rss / 1024 / 1024 / 1024))
    print("max GPU memory allocated: {:.4f} GB".format(max_memory_allocated))


def test_inference(batch_size=1, prompt_seq_len=1, max_full_seq_len=10, loop=1, data_type=torch.float16):
    torch.set_printoptions(precision=6, sci_mode=False)
    torch.set_grad_enabled(False)
    set_random_seed(1)

    tokenizer = LlamaTokenizer.from_pretrained(model_dir)
    np.random.seed(100)
    input_full = np.random.randint(2000, 3000, prompt_seq_len * batch_size, dtype='int64')
    input_inc = np.random.randint(2000, 3000, 1 * batch_size, dtype='int64')
    
    input_full_decoder = torch.from_numpy(input_full).long().reshape(batch_size, prompt_seq_len).cuda()
    input_inc_decoder = torch.from_numpy(input_inc).long().reshape(batch_size, 1).cuda()
    print(input_full_decoder)

    attention_mask = None
    config = AutoConfig.from_pretrained(model_dir)
    free_in_GB = int(torch.cuda.mem_get_info()[0]/1024**3)
    max_memory = f'{int(torch.cuda.mem_get_info()[0]/1024**3)}GB'

    n_gpus = torch.cuda.device_count()
    max_memory = {i: max_memory for i in range(n_gpus)}
    # ts_lm_model = AutoModelForCausalLM.from_pretrained(model_dir,load_in_8bit=True,config=config)
    ts_lm_model = AutoModelForCausalLM.from_pretrained(
        model_dir,
        device_map='cuda:0',
        torch_dtype=data_type,
        max_memory=max_memory
    )

    # for k, v in ts_lm_model.state_dict().items():
    #     print(k, v.shape)
    # print(ts_model.config)

    # ts generate
    '''

    ts_model = ts_lm_model.model
    # generate_ids = ts_model.generate(input_ids, max_length=64)
    # warm up
    for i in range(loop):
        res_ts = ts_model(input_ids=input_full_decoder)
        # print(res_ts)
    # profile
    torch.cuda.synchronize()
    t1 = time.perf_counter()
    for i in range(loop):
        input_ids = input_full_decoder
        past_key_values = None
        for j in range(max_full_seq_len - prompt_seq_len):
            with torch.no_grad():
                res_ts = ts_model(input_ids=input_ids, past_key_values=past_key_values, attention_mask=attention_mask, use_cache=True)
            print("step: ", j, " ts output: ", res_ts.last_hidden_state.reshape(-1)[:12800:640])
            past_key_values = res_ts.past_key_values
            input_ids = input_inc_decoder
    torch.cuda.synchronize()
    t2 = time.perf_counter()
    time_ts = t2 - t1
    print('Time for Transformers: ', time_ts / loop)

    ts_model.cpu()
    '''

    # eet generate
    eet_model = EETLlamaModel.from_torch(ts_lm_model, max_batch=batch_size, max_prompt_seq_len=prompt_seq_len,
                                         max_full_seq_len=max_full_seq_len, data_type=data_type, model_attr='model')
    # profile
    torch.cuda.synchronize()
    t1 = time.perf_counter()
    for i in range(loop):
        input_ids = input_full_decoder
        first_pass = True
        for j in range(max_full_seq_len - prompt_seq_len):
            with torch.no_grad():
                res_eet = eet_model(input_ids=input_ids, first_pass=first_pass, attention_mask=attention_mask)
            # print("step: ", j, " eet output: ", res_eet.reshape(-1)[:12800:640])
            if first_pass:
                first_pass = False
            input_ids = input_inc_decoder

    torch.cuda.synchronize()
    t2 = time.perf_counter()
    time_eet = t2 - t1
    print('Time for eet: ', time_eet / loop)

    print('*****************************')
    print(batch_size, '\t', prompt_seq_len, '\t', max_full_seq_len, '\t', data_type)
    max_memory_allocated = torch.cuda.max_memory_allocated() / 1e9
    print('当前进程号: {}, 内存使用：{:.4f} GB'.format(os.getpid(), psutil.Process(os.getpid()).memory_info().rss / 1024 / 1024 / 1024))
    print("max GPU memory allocated: {:.4f} GB".format(max_memory_allocated))

def test_pytorch(batch_size=1, prompt_seq_len=1, max_full_seq_len=10, loop=10, data_type=torch.float16):
    torch.set_printoptions(precision=3, sci_mode=False)
    torch.set_grad_enabled(False)
    set_random_seed(1)

    tokenizer = LlamaTokenizer.from_pretrained(model_dir)
    
    input_full = np.random.randint(1000, 9000, prompt_seq_len * batch_size, dtype='int64')
    input_inc = np.random.randint(1000, 9000, 1 * batch_size, dtype='int64')

    input_full_decoder = torch.from_numpy(input_full).long().reshape(batch_size, prompt_seq_len).cuda()
    input_inc_decoder = torch.from_numpy(input_inc).long().reshape(batch_size, 1).cuda()

    attention_mask = None    
    ts_lm_model = LlamaForCausalLM.from_pretrained(model_dir)

    # for k, v in ts_model.state_dict().items():
    #     print(k, v.shape)
    # print(ts_model.config)

    # ts generate
    ts_model = ts_lm_model.to(data_type).model.cuda()
    # generate_ids = ts_model.generate(input_ids, max_length=64)
    # warm up
    for i in range(loop):
        res_ts = ts_model(input_ids=input_full_decoder)

    # profiling
    torch.cuda.synchronize()
    t1 = time.perf_counter()
    for i in range(loop):
        input_ids = input_full_decoder
        past_key_values = None
        for j in range(max_full_seq_len - prompt_seq_len):
            with torch.no_grad():
                res_ts = ts_model(input_ids=input_ids, past_key_values=past_key_values, attention_mask=attention_mask, use_cache=True)
            # print("step: ", j, " ts output: ", res_ts.last_hidden_state.reshape(-1)[:1280:64])
            past_key_values = res_ts.past_key_values
            input_ids = input_inc_decoder
    torch.cuda.synchronize()
    t2 = time.perf_counter()
    time_ts = t2 - t1
    print('Time for Transformers: ', time_ts)
    print('*****************************')
    print(batch_size, '\t', prompt_seq_len, '\t', max_full_seq_len, '\t', data_type)
    max_memory_allocated = torch.cuda.max_memory_allocated() / 1e9
    print('当前进程号: {}, 内存使用：{:.4f} GB'.format(os.getpid(), psutil.Process(os.getpid()).memory_info().rss / 1024 / 1024 / 1024))
    print("max GPU memory allocated: {:.4f} GB".format(max_memory_allocated))

def test_output():
    torch.set_grad_enabled(False)
    set_random_seed(1)
    torch.set_printoptions(precision=6, sci_mode=False)
    tokenizer = LlamaTokenizer.from_pretrained(model_dir)
    config = AutoConfig.from_pretrained(model_dir)
    prompt = "你好，介绍一下你自己?"
    inputs = tokenizer(prompt, return_tensors="pt")
    input_ids = inputs["input_ids"].cuda()

    # transformer
    ts_lm_model = LlamaForCausalLM.from_pretrained(model_dir, config=config)    
    ts_model = ts_lm_model.to(torch.float16).model.cuda()
    
    with torch.no_grad():
        res_ts = ts_model(input_ids=input_ids, past_key_values=None, attention_mask=None, use_cache=True)
        print("ts full output: ", res_ts.last_hidden_state.reshape(-1)[:32])
        res_ts_token = torch.argmax(res_ts.last_hidden_state[:, -1, :], dim=-1)
        input_inc_ids = torch.tensor([[res_ts_token]]).cuda()
        res_ts_inc = ts_model(input_ids=input_inc_ids, past_key_values=res_ts.past_key_values, attention_mask=None, use_cache=True)
        print("ts inc output: ", res_ts_inc.last_hidden_state.reshape(-1)[:32])

    # eet
    eet_model = EETLlamaModel.from_torch(ts_lm_model, max_batch=1, max_prompt_seq_len=1024,
                                         max_full_seq_len=2048, data_type=torch.int8, model_attr='model', is_int8=True)
    
    with torch.no_grad():
        res_eet = eet_model(input_ids=input_ids, first_pass=True, attention_mask=None)
        print("eet full output: ", res_eet.reshape(-1)[:32])
        res_eet_token = torch.argmax(res_eet[:, -1, :], dim=-1)
        input_inc_ids = torch.tensor([[res_eet_token]]).cuda()
        res_eet_inc = eet_model(input_ids=input_inc_ids, first_pass=False, attention_mask=None)
        print("eet inc output: ", res_eet_inc.reshape(-1)[:32])


if __name__ == "__main__":
    # test_inference(batch_size=1, prompt_seq_len=100, max_full_seq_len=101, loop=4, data_type=torch.float16)
    # test_pytorch(batch_size=1, prompt_seq_len=100, max_full_seq_len=101, loop=4, data_type=torch.float16)

    # test()
    test_eet()
    # test_output()

