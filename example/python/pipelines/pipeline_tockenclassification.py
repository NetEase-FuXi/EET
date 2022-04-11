import torch
from eet import pipeline

classifier = pipeline("token-classification",data_type = torch.float16)

out = classifier(["My name is Sarah and I live in London and I like"])
print(out)
