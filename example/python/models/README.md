## 运行BERT-适配transformers
[bert_transformers_example](bert_transformers_example.py)
```bash
$ cd EET/example/python/models
$ python bert_transformers_example.py
```

## 运行GPT2-适配transformers
[gpt2_transformers_example](gpt2_transformers_example.py)
```bash
$ cd EET/example/python/models
$ python gpt2_transformers_example.py
```

## 运行GPT2-适配Fairseq
[gpt2_fairseq_example](gpt2_fairseq_example.py)
```bash
$ cd EET/example/python/models
& #下载EET提供的fairseq gpt模型，该模型是随机参数生成的，文本生成结果可能不正确，只用于做demo性能测试以及使用演示
$ wget https://github.com/NetEase-FuXi/EET/releases/download/v1.0.0/resource.zip
$ unzip resource.zip
$ mv resource ../
$ python gpt2_fairseq_example.py
```

其他模型以此类推即可，其中gpt2_from_buffer_example是随机参数为了测试性能。


## LLM测试效果 

Baichuan-7B
```python
# pytorch output
output_str = '解释一下“温故而知新”的意思。\n温故而知新,出自《论语》,意思是温习学过的知识,从而得到新的理解和体会。也指回忆过去,能更好地认识现在。\n\n扩展资料:\n温故而知新,出自《论语》'

# eet output
output_str = '解释一下“温故而知新”的意思。\n温故而知新,出自《论语》,意思是温习学过的知识,从而得到新的理解和体会。也指回忆过去,能更好地认识现在。\n\n扩展资料:\n温故而知新,出自《论语》'
```

## PPL Test

test on WikiText-v2 test dataset
| model | quant mode | ppl |
| --- | :---: | :---: |
| llama-7b | torch(fp16) | 5.677 |
| llama-7b | eet(fp16) | 5.676 |
| llama-7b | eet(int8) | 5.682 |

| model | quant mode | ppl |
| --- | :---: | :---: |
| baichuan-7b | torch(fp16) | 6.030 |
| baichuan-7b | eet(fp16) | 6.036 |
| baichuan-7b | eet(int8) | 6.642 |
