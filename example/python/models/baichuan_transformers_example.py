import os
import time
import psutil
import random
import torch
import numpy as np
from torch.nn.parameter import Parameter
from transformers import AutoTokenizer, AutoConfig, AutoModelForCausalLM
from eet.transformers.modeling_baichuan import convert_baichuan_weights

model_path = "/root/project/huggingface/baichuan2-13b/"
int8_ckpt_path = "/root/project/huggingface/baichuan2-13b/eet_baichuan_int8_layer0.pt"

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

    eet_baichuan_dict = convert_baichuan_weights(ts_model.state_dict())
    print("*******************after quant*****************")
    for k, v in eet_baichuan_dict.items():
        print(k, v.shape)
    torch.save(eet_baichuan_dict, "/root/project/huggingface/baichuan2-13b/eet_baichuan_int8.pt")


def test_pytorch(batch_size=1, prompt_seq_len=1, max_new_tokens=1, loop=1, data_type=torch.float16):
    torch.set_printoptions(precision=3, sci_mode=False)
    torch.set_grad_enabled(False)
    set_random_seed(1)

    tokenizer = AutoTokenizer.from_pretrained(model_path, trust_remote_code=True, use_fast=False)
    config = AutoConfig.from_pretrained(model_path, trust_remote_code=True)
    config.num_hidden_layers = 1
    ts_model = AutoModelForCausalLM.from_pretrained(model_path, config=config, trust_remote_code=True, torch_dtype=data_type)

    ts_model.to("cuda:0")
    num_params = sum(p.numel() for p in ts_model.parameters())
    print("num_params: ", num_params)
    for k, v in ts_model.state_dict().items():
        print(k, v.shape)
    print(ts_model.config)

    attention_mask = None    
    kwargs = {
        "max_new_tokens": int(max_new_tokens),
        "do_sample": bool(False),
        # "temperature": float(0.75),
        "top_k": int(50),
        # "top_p": float(0.7),
        "use_cache": bool(True),
    }
    input_ids = torch.randint(1000, 8000, (batch_size, prompt_seq_len), dtype=torch.long, device='cuda:0')
    kwargs["inputs"] = input_ids.to("cuda:0")

    # warm up
    # for i in range(loop):
    #     res_ts = ts_model(input_ids=input_ids)
    # warm up
    generate_ids = ts_model.generate(**kwargs)

    print('*****************************')
    print(batch_size, '\t', prompt_seq_len, '\t', max_new_tokens, '\t', data_type)
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
    config.num_hidden_layers = 1
    
    attention_mask = None
    # with open(int8_ckpt_path, "rb") as f:
    #     baichuan_dict = torch.load(f, map_location=torch.device("cpu"))
    # eet_model = EETBaichuanModel.from_torch(baichuan_dict, config, max_batch=batch_size, max_prompt_seq_len=prompt_seq_len,
    #                                                 max_full_seq_len=prompt_seq_len+max_new_tokens+1, data_type=torch.float16)
    
    eet_model = EETBaichuanForCausalLM.from_pretrained(int8_ckpt_path, config, max_batch=batch_size, max_prompt_seq_len=prompt_seq_len,
                                                       max_full_seq_len=prompt_seq_len+max_new_tokens+1, data_type=torch.float16)

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
    #     res_ts = eet_model(input_ids=input_full_decoder)
        # print(res_ts)
    # profile
    torch.cuda.synchronize()
    t1 = time.perf_counter()
    for i in range(loop):
        input_ids = input_full_decoder
        self_past_key_values_length = 0
        first_pass = True
        for j in range(max_new_tokens):
            with torch.no_grad():
                res_eet = eet_model(input_ids=input_ids, first_pass=first_pass, attention_mask=attention_mask, self_past_key_values_length=self_past_key_values_length)
            # print("step: ", j, " eet output: ", res_eet.reshape(-1)[:12800:640])
            self_past_key_values_length += input_ids.shape[1]
            if first_pass:
                first_pass = False
            input_ids = input_inc_decoder
    torch.cuda.synchronize()
    t2 = time.perf_counter()
    time_eet = t2 - t1
    # print("batch_size: {}, prompt_length: {}, max_new_tokens: {}, lantency: {:.4f} s".format(batch_size, prompt_seq_len, max_new_tokens, time_eet / loop))
    # print('Time for eet: ', time_eet / loop)
    max_memory_allocated = torch.cuda.max_memory_allocated() / 1e9
    print('当前进程号: {}, 内存使用：{:.4f} GB'.format(os.getpid(), psutil.Process(os.getpid()).memory_info().rss / 1024 / 1024 / 1024))
    print("max GPU memory allocated: {:.4f} GB".format(max_memory_allocated))


def test_eet_generate():
    from eet import EETBaichuanModel, EETBaichuanForCausalLM

    torch.set_printoptions(precision=6, sci_mode=False)
    torch.set_grad_enabled(False)
    set_random_seed(1)
    tokenizer = AutoTokenizer.from_pretrained(model_path, trust_remote_code=True, use_fast=False)
    config = AutoConfig.from_pretrained(model_path, trust_remote_code=True)

    attention_mask = None
    eet_ckpt_path = "/root/project/huggingface/baichuan2-13b/eet_baichuan_int8.pt"
    eet_model = EETBaichuanForCausalLM.from_pretrained(eet_ckpt_path, config, max_batch=4, max_prompt_seq_len=100,
                                                       max_full_seq_len=1024, data_type=torch.float16)

    text = '中国的首都在'
    # text = '晚上睡不着应该怎么办'
    kwargs = {
        "max_new_tokens": int(50),
        # "min_new_tokens": int(50),
        "do_sample": bool(False),
        # "temperature": float(0.75),
        "top_k": int(50),
        # "top_p": float(0.7),
        "use_cache": bool(True),
    }
    inputs = tokenizer(str(text), return_tensors='pt')
    kwargs["inputs"] = inputs.input_ids.to('cuda').repeat(2, 1)

    # generate
    generate_ids = eet_model.generate(**kwargs)
    outputs_str = tokenizer.batch_decode(
        generate_ids, 
        skip_special_tokens=True, 
        clean_up_tokenization_spaces=False)
    print("outputs_str", outputs_str)

if __name__ == "__main__":
    # save_dict()
    test_eet_generate()
    # test_eet_inference(batch_size=1, prompt_seq_len=4, max_new_tokens=2, loop=0)
    # test_pytorch(batch_size=1, prompt_seq_len=4, max_new_tokens=2, loop=1, data_type=torch.float16)
