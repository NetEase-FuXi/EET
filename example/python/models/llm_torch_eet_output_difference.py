import os
import time
import psutil
import random
import torch
import numpy as np
from tqdm import tqdm
from torch.nn.parameter import Parameter
from transformers import AutoTokenizer, AutoConfig, AutoModelForCausalLM, T5Model, T5Tokenizer
from datasets import load_dataset
from eet.transformers.modeling_baichuan import convert_baichuan_weights
from eet import EETLlamaModel, EETLlamaForCausalLM, EETBaichuanModel, EETBaichuanForCausalLM
from eet.transformers.modeling_t5 import EETT5Model

import matplotlib.pyplot as plt

model_dir = "/root/download/baichuan2-7b-base"
# model_dir = "/root/download/llama-7b"

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

    eet_model = EETLlamaForCausalLM.from_pretrained(config, config, max_batch=1, max_prompt_seq_len=1024,
                                                     max_full_seq_len=2048, data_type=torch.float16, model_attr='model', is_int8=False)
    # generate
    with torch.no_grad():
        res_eet = eet_model(input_ids=input_ids, past_key_values=None, attention_mask=None)
        print("eet full output: ", res_eet.reshape(-1)[:32])

def test_baichuan_output():
    torch.set_grad_enabled(False)
    set_random_seed(1)
    torch.set_printoptions(precision=6, sci_mode=False)
    tokenizer = AutoTokenizer.from_pretrained(model_dir, use_fast=False, trust_remote_code=True)
    config = AutoConfig.from_pretrained(model_dir, use_fast=False, trust_remote_code=True)
    prompt = "解释一下“温故而知新”"
    inputs = tokenizer(prompt, return_tensors="pt")
    input_ids = inputs["input_ids"].cuda()

    # transformer
    ts_lm_model = AutoModelForCausalLM.from_pretrained(model_dir, config=config, use_fast=False, trust_remote_code=True).half().cuda()
    # generate
    with torch.no_grad():
        res_ts = ts_lm_model(input_ids=input_ids, past_key_values=None, attention_mask=None, use_cache=True)
        res_ts_last = res_ts.logits[:, -1:, :]
        print("ts full output: ", res_ts_last.reshape(-1)[:32])
    
    ts_lm_model = ts_lm_model.cpu()
    del ts_lm_model

    baichuanmodel = EETBaichuanForCausalLM.from_pretrained(model_dir, config, max_batch=1, max_prompt_seq_len=1024,
                                                           max_full_seq_len=2048, data_type=torch.float16)

    # generate
    with torch.no_grad():
        res_eet = baichuanmodel(input_ids=input_ids, past_key_values=None, attention_mask=None, use_cache=True)
        res_eet_last = res_eet.logits[:, -1:, :]
        print("eet full output: ", res_eet_last.reshape(-1)[:32])
    
    res_ts_last = res_ts_last.cpu().numpy()
    res_eet_last = res_eet_last.cpu().numpy()
    print(f"abs diff: {np.mean(np.abs(res_ts_last - res_eet_last)):.3f} +- {np.std(np.abs(res_ts_last - res_eet_last)):.3f}")
    print(f"percent diff: {np.mean(np.abs(res_ts_last - res_eet_last) / np.abs(res_ts_last) * 100):.3f}% +- {np.std(np.abs(res_ts_last - res_eet_last) / np.abs(res_ts_last) * 100):.3f}")
    print(f"abs square error: {np.mean(np.abs(res_ts_last - res_eet_last)**2):.3f} +- {np.std(np.abs(res_ts_last - res_eet_last)**2):.3f}")

    # 画图
    plt.figure(0, dpi=300)
    x = np.arange(len(res_ts_last.reshape(-1)[::90]))
    plt.scatter(x, res_ts_last.reshape(-1)[::90], label="ts", alpha=0.7)
    plt.scatter(x, res_eet_last.reshape(-1)[::90], label="eet", alpha=0.7)
    plt.xlabel("index")
    plt.ylabel("value")
    plt.legend()
    plt.savefig('./baichuan_output.png')
    print('save baichuan_output.png')


def test_T5_output():
    torch.set_grad_enabled(False)
    set_random_seed(1)
    torch.set_printoptions(precision=6, sci_mode=False)
    seq_len = 512
    batch_size = 4

    tokenizer = AutoTokenizer.from_pretrained("/root/download/flan-t5-small/", use_fast=False, trust_remote_code=True)
    input_full = np.random.randint(1000, 9000, seq_len * batch_size, dtype='int64')
    input_inc = np.random.randint(1000, 9000, 1 * batch_size, dtype='int64')

    input_full_decoder = torch.from_numpy(input_full).long().reshape(batch_size, seq_len).cuda()
    input_inc_decoder = torch.from_numpy(input_inc).long().reshape(batch_size, 1).cuda()

    attention_mask = None

    # compute per sample length EET增量推理需要输入encoder out的真实句长
    encoder_seq_len = torch.tensor([seq_len] * batch_size).int().cuda()
    if attention_mask is not None:
        pre_padding_len = torch.sum(1 - attention_mask, 1).int().cuda()
        encoder_seq_len = encoder_seq_len - pre_padding_len

    # load transformers model
    ts_model = T5Model.from_pretrained("/root/download/flan-t5-small/").cuda()
    ts_model = ts_model.half()
    
    torch.cuda.synchronize()
    input_ids = input_full_decoder
    past_key_values = None
    encoder_outputs = None
    with torch.no_grad():
        res_ts = ts_model(input_ids=input_ids, decoder_input_ids=input_ids, past_key_values=past_key_values, attention_mask=attention_mask, encoder_outputs=encoder_outputs)
        past_key_values = res_ts.past_key_values
        input_ids = input_inc_decoder
        encoder_outputs = (res_ts.encoder_last_hidden_state, )
        print("ts full output: ", res_ts.last_hidden_state.reshape(-1)[:32])
    res_ts_last = res_ts.last_hidden_state[:, -1:, :].reshape(-1)[::32]
    
    # load eet model
    max_seq_len = 1024
    eet_model = EETT5Model.from_pretrained("/root/download/flan-t5-small/", batch_size, max_prompt_seq_len=seq_len, max_full_seq_len=max_seq_len, data_type=torch.float16)
    with torch.no_grad():
        self_past_key_values_length = 0
        first_pass = True
        res_eet = eet_model(input_ids=input_ids, decoder_input_ids=input_ids, attention_mask=attention_mask, encoder_seq_length=encoder_seq_len, first_pass=first_pass, self_past_key_values_length=self_past_key_values_length)
        print("eet full output: ", res_eet.reshape(-1)[:32])
    res_eet_last = res_eet[:, -1:, :].reshape(-1)[::32]

    res_ts_last = res_ts_last.cpu().numpy()
    res_eet_last = res_eet_last.cpu().numpy()

    print(f"abs diff: {np.mean(np.abs(res_ts_last - res_eet_last)):.3f} +- {np.std(np.abs(res_ts_last - res_eet_last)):.3f}")
    print(f"percent diff: {np.mean(np.abs(res_ts_last - res_eet_last) / np.abs(res_ts_last) * 100):.3f}% +- {np.std(np.abs(res_ts_last - res_eet_last) / np.abs(res_ts_last) * 100):.3f}")
    print(f"abs square error: {np.mean(np.abs(res_ts_last - res_eet_last)**2):.3f} +- {np.std(np.abs(res_ts_last - res_eet_last)**2):.3f}")

    # 画图
    plt.figure(0, dpi=300)
    x = np.arange(len(res_ts_last))
    plt.scatter(x, res_ts_last, label="ts", alpha=0.7)
    plt.scatter(x, res_eet_last, label="eet", alpha=0.7)
    plt.xlabel("index")
    plt.ylabel("value")
    plt.legend()
    plt.savefig('./t5_output.png')
    print('save t5_output.png')


def ppl_eval(tokenizer, model):
    seq_len = 2048

    # 1. WikiText
    testdata = load_dataset('wikitext', 'wikitext-2-raw-v1', split='test')
    prompt = "\n\n".join(testdata['text'])

    # 2. Chinese dataset
    # dataset = load_dataset("json", data_files={"test": "/data/test_chinese.json"})
    # prompt = "".join(dataset["test"]["text"])

    test_ids = tokenizer(prompt, return_tensors='pt').input_ids[0]
    test_ids_batch = []
    nsamples = test_ids.numel() // seq_len
    for i in range(nsamples):
        batch = test_ids[(i * seq_len):((i + 1) * seq_len)]
        test_ids_batch.append(batch.unsqueeze(0).cuda())

    nlls = []
    loss_fct = torch.nn.CrossEntropyLoss(reduction="none")
    with torch.no_grad():
        for batch_idx in tqdm(range(len(test_ids_batch))):
            input_id = test_ids_batch[batch_idx]
            output = model(input_ids=input_id, past_key_values=None, attention_mask=None, use_cache=False)
            lm_logits = output.logits
        
            # Shift so that tokens < n predict n
            shift_logits = lm_logits[..., :-1, :].contiguous()
            shift_labels = input_id[..., 1:].contiguous()
            # Flatten the tokens
            shift_logits = shift_logits.squeeze(0)
            shift_labels = shift_labels.squeeze(0)
            # Enable model parallelism
            shift_labels = shift_labels.to(shift_logits.device)
            loss = loss_fct(shift_logits, shift_labels)
            nlls.append(loss)
    #print(torch.cat(nlls, dim=-1).mean())
    ppl = np.exp(torch.cat(nlls, dim=-1).mean().item())
    print("ppl: ", ppl)
    return ppl

def test_torch_ppl():
    tokenizer = AutoTokenizer.from_pretrained(model_dir, trust_remote_code=True)
    config = AutoConfig.from_pretrained(model_dir, trust_remote_code=True)
    # transformer
    ts_lm_model = AutoModelForCausalLM.from_pretrained(model_dir, config=config, trust_remote_code=True)
    ts_model = ts_lm_model.to(torch.float16).cuda()
    ppl_eval(tokenizer, ts_model)

def test_eet_ppl():
    tokenizer = AutoTokenizer.from_pretrained(model_dir, trust_remote_code=True)
    config = AutoConfig.from_pretrained(model_dir, trust_remote_code=True)

    # baichuan
    ts_model = AutoModelForCausalLM.from_pretrained(model_dir, config=config, trust_remote_code=True).to(torch.float16).cuda()
    ppl_eval(tokenizer, ts_model)
    # model_ckpt_dir = "/root/download/baichuan2-7b-base/eet_baichuan_int8.pt"
    # baichuanmodel = EETBaichuanForCausalLM.from_pretrained(model_ckpt_dir, config, max_batch=1, max_prompt_seq_len=2048,
    #                                                        max_full_seq_len=4096, data_type=torch.int8)
    # ppl_eval(tokenizer, baichuanmodel)

    # llama
    # ts_model = AutoModelForCausalLM.from_pretrained(model_dir, config=config, trust_remote_code=True).to(torch.float16).cuda()
    # ppl_eval(tokenizer, ts_model)
    # eet_model = EETLlamaForCausalLM.from_pretrained(model_dir, config, max_batch=1, max_prompt_seq_len=2048,
    #                                                 max_full_seq_len=4096, data_type=torch.float16)
    # ppl_eval(tokenizer, eet_model)


if __name__ == '__main__':
    # set_random_seed(1)
    # test_llama_output()

    # test_baichuan_output()
    # test_T5_output()

    # 计算PPL
    test_eet_ppl()
