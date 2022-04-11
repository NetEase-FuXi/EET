import torch
from eet import pipeline
from eet import EETRobertaForMaskedLM
from transformers import RobertaTokenizer,RobertaForMaskedLM

us_model_api = False
us_pipeline = True
max_batch_size = 1
model_path = 'roberta-base'
data_type = torch.float16

input = ["My <mask> is Sarah and I live in London"]

if us_pipeline:
    nlp = pipeline("fill-mask",model = model_path,data_type = data_type,max_batch_size = max_batch_size)
    out = nlp(input)
    print('pipeline out:',out)

if us_model_api:
    tokenizer = RobertaTokenizer.from_pretrained(model_path)
    eet_roberta_model = EETRobertaForMaskedLM.from_pretrained(model_path,max_batch = max_batch_size,data_type = data_type)

    # first step: tokenize
    model_inputs = tokenizer(input,return_tensors = 'pt')
    masked_index = torch.nonzero(model_inputs['input_ids'][0] == tokenizer.mask_token_id, as_tuple=False).squeeze(-1)
    # second step: predict
    prediction_scores = eet_roberta_model(model_inputs['input_ids'].cuda(),attention_mask = model_inputs['attention_mask'])
    
    # third step: argmax
    predicted_index = torch.argmax(prediction_scores.logits[0, masked_index]).item()
    predicted_token = tokenizer.convert_ids_to_tokens(predicted_index)

    print('model out:',predicted_token)
