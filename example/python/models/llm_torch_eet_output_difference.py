import os
import time
import psutil
import random
import torch
import numpy as np
from torch.nn.parameter import Parameter
from transformers import AutoTokenizer, AutoConfig, AutoModelForCausalLM
from eet.transformers.modeling_baichuan import convert_baichuan_weights
from eet import EETLlamaModel, EETLlamaForCausalLM


model_dir = "/root/download/baichuan-7b-chat"

def set_random_seed(seed):
    random.seed(seed)
    torch.manual_seed(seed)
    torch.backends.cudnn.deterministic = True
    torch.cuda.manual_seed_all(seed)
    np.random.seed(seed)

def test_llama_output():
    torch.set_grad_enabled(False)
    set_random_seed(1)
    torch.set_printoptions(precision=6, sci_mode=False)
    tokenizer = AutoTokenizer.from_pretrained(model_dir)
    config = AutoConfig.from_pretrained(model_dir)
    prompt = "中国的首都在哪里?"
    inputs = tokenizer(prompt, return_tensors="pt")
    input_ids = inputs["input_ids"].cuda()

    # transformer
    ts_lm_model = AutoModelForCausalLM.from_pretrained(model_dir, config=config)
    ts_model = ts_lm_model.to(torch.float16).model.cuda()
    # generate
    with torch.no_grad():
        res_ts = ts_model(input_ids=input_ids, past_key_values=None, attention_mask=None, use_cache=True)
        print("ts full output: ", res_ts.last_hidden_state.reshape(-1)[:32])

    model_dict = ts_lm_model.state_dict()
    lm_head_dict = {}
    llama_dict = {}
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
    # generate
    with torch.no_grad():
        res_eet = eet_model(input_ids=input_ids, past_key_values=None, attention_mask=None, use_cache=True)
        print("eet full output: ", res_eet.reshape(-1)[:32])


if __name__ == '__main__':
    # set_random_seed(1)
    test_llama_output()


