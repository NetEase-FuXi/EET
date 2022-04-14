import torch
from eet import pipeline
us_fairseq_model = True
us_thansformers_model = True


if us_fairseq_model:
    # 参考Readme,首先需要生成vocab.txt和config.json文件,并且你的pt文件名必须是checkpoint_best.pt，这个将会在后续版本做优化
    nlp = pipeline("text-generation",model = '../resource', data_type = torch.float16)
    out = nlp(["我 叫 小 天 ，这 是","这 里 是 中 国 ，不 是"])
    print(out)

if us_thansformers_model:
    nlp = pipeline("text-generation", data_type = torch.float16)
    out = nlp("My name is")
    print(out)
