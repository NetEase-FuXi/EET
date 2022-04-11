from eet import pipeline
import torch
import time
nlp = pipeline("question-answering",data_type = torch.float32)
QA_input = {
    'question': 'Why is model conversion important?',
    'context': 'The option to convert models between FARM and transformers gives freedom to the user and let people easily switch between frameworks.'
}
out = nlp(QA_input)
print(out)
