import torch
from eet import pipeline

pipe = pipeline("text-classification",data_type = torch.float16)
out = pipe(["This restaurant is awesome","This restaurant is very bad"])
print(out)

