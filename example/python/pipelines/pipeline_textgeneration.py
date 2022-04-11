import torch
from eet import pipeline
us_fairseq_model = True
us_thansformers_model = True

if us_fairseq_model:
    nlp = pipeline("text-generation",model = '/root/3090_project/git/fairseq/checkpoint', data_type = torch.float16)
    out = nlp(["我 叫 小 天 ，这 是","这 里 是 中 国 ，不 是"])
    print(out)

if us_thansformers_model:
    nlp = pipeline("text-generation", data_type = torch.float16)
    out = nlp("My name is")
    print(out)
