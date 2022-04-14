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
$ wget https://github.com/NetEase-FuXi/EET/releases/download/EET_V0.0.1_fairseq0.10.0_transformers3.5.0/resource.zip
$ unzip resource.zip
$ mv resource ../
$ python gpt2_fairseq_example.py
```

其他模型以此类推即可，其中gpt2_from_buffer_example是随机参数为了测试性能。