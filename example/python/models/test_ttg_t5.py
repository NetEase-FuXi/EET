import torch
import numpy as np
from torch.nn.parameter import Parameter
from transformers import T5Model, T5Tokenizer, BartModel, T5ForConditionalGeneration
from eet.transformers.modeling_t5 import EETT5Model, EETT5ForConditionalGeneration
from PIL import Image
import requests
import time

# using_half = True
# batch_size = 4
# seq_len = 16
# loop = 1
# model_dir = "t5-small"
# model_dir = "/root/data/models/0627_test_t5/"
# model_dir = "/root/data/models/0627_test_gated_gelu/"
model_dir = "/root/data/models/test_ttg_t5/"

device = "cuda" if torch.cuda.is_available() else "cpu"

def test(batch_size=4, seq_len=8, max_seq_len=1024, using_half=True, using_eet=True, using_ts=True, loop=100):
    torch.set_printoptions(precision=3, sci_mode=False)
    torch.set_grad_enabled(False)
    # 输入数据构造，实际业务输入应该是tokens
    # tokenizer = T5Tokenizer.from_pretrained("t5-small")
    # inputs = tokenizer("Hello, my dog is cute", return_tensors="pt")
    # fake data
    input_full = np.random.randint(1000, 9000, seq_len * batch_size, dtype='int64')
    input_inc1 = np.random.randint(1000, 9000, 1 * batch_size, dtype='int64')
    input_inc2 = np.random.randint(1000, 9000, 1 * batch_size, dtype='int64')

    input_full_decoder = torch.from_numpy(input_full).long().reshape(batch_size, seq_len).cuda()
    input_inc_decoder1 = torch.from_numpy(input_inc1).long().reshape(batch_size, 1).cuda()
    input_inc_decoder2 = torch.from_numpy(input_inc2).long().reshape(batch_size, 1).cuda()

    data_type = torch.float16 if using_half else torch.float32
    attention_mask = None

    # compute per sample length EET增量推理需要输入encoder out的真实句长
    encoder_seq_len = torch.tensor([seq_len] * batch_size).int().cuda()
    if attention_mask is not None:
        pre_padding_len = torch.sum(1 - attention_mask, 1).int().cuda()
        encoder_seq_len = encoder_seq_len - pre_padding_len

    if using_ts:
        ts_model = T5Model.from_pretrained(model_dir)
        ts_model = ts_model.half().cuda() if using_half else ts_model.cuda()
        print(ts_model.config)
        
        past_key_values = None
        encoder_outputs = None
        # warm up
        # for i in range(10):
        #     with torch.no_grad():
        #         res_ts_full = ts_model(input_ids=input_full_decoder, decoder_input_ids=input_full_decoder, past_key_values=past_key_values, attention_mask=attention_mask, encoder_outputs=encoder_outputs)

        torch.cuda.synchronize()
        t3 = time.perf_counter()
        for i in range(loop):
            input_ids = input_full_decoder
            decoder_input_ids = input_inc_decoder1
            past_key_values = None
            encoder_outputs = None
            for j in range(max_seq_len):
                with torch.no_grad():
                    res_ts = ts_model(input_ids=input_ids, decoder_input_ids=decoder_input_ids, past_key_values=past_key_values, attention_mask=attention_mask, encoder_outputs=encoder_outputs)
                # if j == (max_seq_len - 1) and i == (loop - 1):
                #     print(j, ' res ts: ', res_ts.last_hidden_state, ' shape: ', res_ts.last_hidden_state.size())
                past_key_values = res_ts.past_key_values
                input_ids = input_inc_decoder2
                decoder_input_ids = input_inc_decoder2
                encoder_outputs = (res_ts.encoder_last_hidden_state, )
        torch.cuda.synchronize()
        t4 = time.perf_counter()
        time_ts = t4 - t3

    if using_eet:
    # load model
        eet_model = EETT5Model.from_pretrained(model_dir, batch_size, data_type=data_type)

        '''
        first_pass 用于判断生成任务时是否是第一步，也就是是否是在做提示词的推理。true代表在做提示词的推理，false代表在做生成推理
        由于eet不会返回past_key_value，前一步的信息全部在内部做了保存，所以没法通过past_key_value做判断，故增加此参数。
        '''
        torch.cuda.synchronize()
        t1 = time.perf_counter()
        for i in range(loop):
            input_ids = input_full_decoder
            decoder_input_ids = input_inc_decoder1
            self_past_key_values_length = 0
            first_pass = True
            for j in range(max_seq_len):
                res_eet = eet_model(input_ids=input_ids, decoder_input_ids=decoder_input_ids, attention_mask=attention_mask, encoder_seq_length=encoder_seq_len, first_pass=first_pass, self_past_key_values_length=self_past_key_values_length)
                self_past_key_values_length += decoder_input_ids.shape[1]
                # if j == (max_seq_len - 1) and i == (loop - 1):
                #     print(j, ' res eet: ', res_eet, ' shape: ', res_eet.shape)
                if first_pass:
                    first_pass = False
                input_ids = input_inc_decoder2
                decoder_input_ids = input_inc_decoder2
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
    print(batch_size, '\t', seq_len, '\t', max_seq_len, '\t', fp)

def test_text_generate():
    min_length = 30
    max_length = 100
    torch.set_grad_enabled(False)
    model_dir = "/root/data/models/test_ttg_t5/"
    
    tokenizer = AutoTokenizer.from_pretrained(model_dir)
    model = T5ForConditionalGeneration.from_pretrained(model_dir, from_flax=False).to('cuda:0')
    eet_model = EETT5ForConditionalGeneration.from_pretrained(model_dir, max_batch=10, data_type=torch.float16)
    
    input_str = "少侠路过码头时，一位船夫哽咽着向少侠求救：“少侠，我两年前出海时答应过儿子会给他带一把铁木剑，可船靠岸时那把剑不小心掉水里了，我沿着船舷上刻的记号找了许久也没找到， \
    # 这可如何是好？”少侠说：“帮你下水去捞。”<extra_id_0>船夫的儿子获得了铁木剑，十分开心。</s>"
    input_str = input_str.lower().replace(":", '：').replace('“' , '"').replace('”', '"').replace('!', '！').replace('?', '？')
    inputs = tokenizer([input_str] * 10, return_tensors='pt', add_special_tokens=False)
    input_ids = inputs['input_ids'].to(model.device)
    print("input_ids length" , input_ids.size())
    print(input_str)

    # ts
    torch.cuda.synchronize()
    t1 = time.perf_counter()
    bot_outs = model.generate(
        input_ids, 
        top_k=20,
        top_p=0.9,
        do_sample=True,
        temperature=0.88,
        repetition_penalty=1.0,
        no_repeat_ngram_size=7, 
        min_length=min_length, 
        max_length=max_length, 
        bad_words_ids=[[tokenizer.unk_token_id]],
        eos_token_id=21129,
        return_dict_in_generate = True,
        output_scores=True,
        bos_token_id=0, 
        decoder_start_token_id=0
        )#"<extra_id_1>" 如果是150万就不需要这个，训练的时候没加
    torch.cuda.synchronize()
    t2 = time.perf_counter()
    time_ts_full = t2 - t1

    gen_sequences_id = bot_outs.sequences
    # gen_sequences_id = bot_outs
    
    gen_sequences_id = gen_sequences_id[:,1:]#去掉开始的pad
    lengths = torch.sum(gen_sequences_id != tokenizer.pad_token_id, -1)
    print('generate output length: ', gen_sequences_id.size(), '\t', lengths)    
    gene_outs = tokenizer.batch_decode(gen_sequences_id, skip_special_tokens=True, clean_up_tokenization_spaces=False)#False
    print('generated results: ', gene_outs)

    scores = bot_outs.scores #tuple of tensor :length, (num, vocab)
    probs = torch.stack(scores, dim=1).softmax(-1)
    gen_probs = torch.gather(probs, 2, gen_sequences_id[:, :, None]).squeeze(-1)#pad的分数都是0
    gen_probs = gen_probs.sum(-1) / lengths
    print(gen_probs)

    #eet
    torch.cuda.synchronize()
    t1 = time.perf_counter()
    eet_gen_sequences_id = eet_model.generate(
        input_ids, 
        top_k=20,
        top_p=0.9,
        do_sample=True,
        temperature=0.88,
        repetition_penalty=1.0,
        no_repeat_ngram_size=7, 
        min_length=min_length, 
        max_length=max_length, 
        bad_words_ids=[[tokenizer.unk_token_id]],
        eos_token_id=21129,
        return_dict_in_generate = True,
        output_scores=True,
        bos_token_id=0, 
        decoder_start_token_id=0
        )#"<extra_id_1>" 如果是150万就不需要这个，训练的时候没加

    torch.cuda.synchronize()
    t2 = time.perf_counter()
    time_eet_full = t2 - t1
    # time
    print("eet full time: ", time_eet_full)
    print("ts full time: ", time_ts_full)
    print(input_str)

    eet_gen_sequences_id = eet_gen_sequences_id[:,1:]#去掉开始的pad
    lengths = torch.sum(eet_gen_sequences_id != tokenizer.pad_token_id, -1)
    print('eet generate output length: ', eet_gen_sequences_id.size(), '\t', lengths)    
    eet_gene_outs = tokenizer.batch_decode(eet_gen_sequences_id, skip_special_tokens=True, clean_up_tokenization_spaces=False)#False
    print(' eet generated results: ', eet_gene_outs)

if __name__ == '__main__':
    test(batch_size=10, seq_len=32, max_seq_len=100,
         using_half=True, using_eet=True, using_ts=True, loop=10)
