import torch
import numpy as np
from torch.nn.parameter import Parameter
# from eet.transformers.modeling_vit import EETViTModel
# from eet.transformers.modeling_clip import EETCLIPModel
from transformers import T5Model, CLIPModel, CLIPProcessor
from eet.transformers.modeling_t5 import EETT5Model
from PIL import Image
import requests
import time

using_half = False
batch_size = 2
seq_len = 4
loop = 1
device = "cuda" if torch.cuda.is_available() else "cpu"

def main():
    torch.set_printoptions(precision=3, sci_mode=False)
    torch.set_grad_enabled(False)

    data_type = torch.float16 if using_half else torch.float32

    ts_model = T5Model.from_pretrained("t5-small").cuda()
    eet_model = EETT5Model.from_pretrained("t5-small", batch_size, seq_len, data_type=data_type)
    cfg = ts_model.config
    print("config: ", cfg)
    print("**************")
    # for k, v in ts_model.state_dict().items():
    #     print(k)
    input_ids = torch.from_numpy(np.ones((4, 9), dtype=np.int64)).cuda()
    decoder_input_ids = torch.from_numpy(np.random.randint(1, 10, 4*9, dtype="int64")).long().reshape(4, 9).cuda()

    outputs = ts_model(input_ids=input_ids, decoder_input_ids=decoder_input_ids)
    last_hidden_states = outputs.last_hidden_state
    print(last_hidden_states.size())

    # eet
    eet_output = eet_model(input_ids=input_ids, first_pass=True, decoder_input_ids=decoder_input_ids)
    print("eet output size: ", eet_output.size())


def test01():
    torch.set_printoptions(precision=3, sci_mode=False)
    torch.set_grad_enabled(False)

    data_type = torch.float32

    ts_model = CLIPModel.from_pretrained("openai/clip-vit-base-patch32").cuda()
    processor = CLIPProcessor.from_pretrained("openai/clip-vit-base-patch32")
    if using_half:
        data_type = torch.float16
        ts_model = ts_model.half()
    
    cfg = ts_model.config
    image = Image.open('/root/data/1.jpg')
    inputs = processor(text=["a photo of a cat", "a photo of a dog"], images=image, return_tensors="pt", padding=True)
    
    pixel_values = inputs['pixel_values'].to(data_type).cuda()
    input_ids = inputs['input_ids'].cuda()
    attention_mask = inputs['attention_mask'].to(data_type).cuda()
    input_ids = torch.from_numpy(np.ones((1, 4), dtype=np.int64)*4).cuda()
    # input_ids = torch.from_numpy(np.random.randint(1000, 9000, seq_len * batch_size, dtype="int64")).reshape(batch_size, seq_len).cuda()

    num_channels = pixel_values.shape[1]
    eet_model = EETCLIPModel.from_pretrained("openai/clip-vit-base-patch32", max_batch=batch_size, num_channels=num_channels, data_type=data_type)
    ts_text_embedding = ts_model.text_model.embeddings
    ts_text_model = ts_model.text_model.encoder.layers
    
    print("************ts text model*****************")
    embed_out = ts_text_embedding(input_ids)
    print('embed_out: ', embed_out)
    encoder_out = ts_text_model[0](embed_out, attention_mask=None, causal_attention_mask=None)
    encoder_out = ts_text_model[1](encoder_out[0], attention_mask=None, causal_attention_mask=None)
    print('encoder out: ', encoder_out)
    

    # res_eet = eet_model(input_ids, pixel_values, attention_mask=attention_mask)
    print('***************eet text model**************')
    eet_text_embedding = eet_model.text_model.embedding
    eet_text_model = eet_model.text_model.encoder.layers
    position_ids = torch.arange(0, cfg.text_config.max_position_embeddings).reshape(1, cfg.text_config.max_position_embeddings).cuda()
    position_ids = position_ids[:, :input_ids.size()[1]]
    token_type_ids = torch.empty(0).long()
    embed_out = eet_text_embedding(input_ids, position_ids, token_type_ids)
    print('embed_out: ', embed_out)
    pre_padding_len = torch.empty(0).long()
    encoder_out = eet_text_model[0](embed_out, pre_padding_len=pre_padding_len, normalize_before=True)
    encoder_out = eet_text_model[1](encoder_out, pre_padding_len=pre_padding_len, normalize_before=True)
    print('encoder out: ', encoder_out)

def test02():
    torch.set_printoptions(precision=3, sci_mode=False)
    torch.set_grad_enabled(False)

    data_type = torch.float32

    ts_model = CLIPModel.from_pretrained("openai/clip-vit-base-patch32").cuda()
    processor = CLIPProcessor.from_pretrained("openai/clip-vit-base-patch32")
    if using_half:
        data_type = torch.float16
        ts_model = ts_model.half()
    
    cfg = ts_model.config
    print("vision config: ", cfg.vision_config)
    print("text_config: ", cfg.text_config)
    print("**************")
    image = Image.open('/root/data/1.jpg')
    inputs = processor(text=["a photo of a cat", "a photo of a a dog", "a photo", "a photo of a a a dog"], images=image, return_tensors="pt", padding=True)
    
    pixel_values = inputs['pixel_values'].to(data_type).cuda()
    input_ids = inputs['input_ids'].cuda()
    attention_mask = inputs['attention_mask'].to(data_type).cuda()
    print("*******attention_mask: ", attention_mask)
    pixel_values = torch.from_numpy(np.ones((1, 3, 224, 224), dtype=np.float32)/2).cuda()
    # input_ids = torch.from_numpy(np.ones((2, 64), dtype=np.int64)*40000).cuda()
    input_ids = np.random.randint(1, 40000, seq_len * batch_size, dtype="int64")
    input_ids = torch.from_numpy(input_ids).reshape(batch_size, seq_len).cuda()
    print("input ids: ", input_ids)
    attention_mask = None

    num_channels = pixel_values.shape[1]
    eet_model = EETCLIPModel.from_pretrained("openai/clip-vit-base-patch32", max_batch=batch_size, num_channels=num_channels, data_type=data_type)
    
    ts_text_model = ts_model.text_model
    eet_text_model = eet_model.text_model

    res_ts = ts_text_model(input_ids, attention_mask=attention_mask)
    print(res_ts.pooler_output.size(), ' res ts text: \n', res_ts.pooler_output)
    res_eet = eet_text_model(input_ids, attention_mask=attention_mask)
    print(res_eet.size(), ' res eet text: \n', res_eet)


def test03():
    torch.set_printoptions(precision=3, sci_mode=False)
    torch.set_grad_enabled(False)

    data_type = torch.float32

    ts_model = CLIPModel.from_pretrained("openai/clip-vit-base-patch32").cuda()
    processor = CLIPProcessor.from_pretrained("openai/clip-vit-base-patch32")
    if using_half:
        data_type = torch.float16
        ts_model = ts_model.half()
    
    cfg = ts_model.config
    # print("config: ", cfg)
    print("vision_config: ", cfg.vision_config)
    print("**************")
    image = Image.open('/root/data/1.jpg')
    inputs = processor(text=["a photo of a cat", "a photo of a a dog", "a photo", "a photo of a a a dog"], images=image, return_tensors="pt", padding=True)
    
    pixel_values = inputs['pixel_values'].to(data_type).cuda()
    input_ids = inputs['input_ids'].cuda()
    attention_mask = inputs['attention_mask'].to(data_type).cuda()
    print("*******attention_mask: ", attention_mask)
    pixel_values = torch.from_numpy(np.ones((1, 3, 224, 224), dtype=np.float32)/2).cuda()
    input_ids = torch.from_numpy(np.ones((1, 5), dtype=np.int64)).cuda()

    num_channels = pixel_values.shape[1]
    eet_model = EETCLIPModel.from_pretrained("openai/clip-vit-base-patch32", max_batch=batch_size, num_channels=num_channels, data_type=data_type)
    
    ts_vision_model = ts_model.vision_model
    eet_vision_model = eet_model.vision_model

    res_ts = ts_vision_model(pixel_values)
    print(res_ts.pooler_output.size(), ' res ts visual: ', res_ts.pooler_output[:, :10])
    res_eet = eet_vision_model(pixel_values)
    print(res_eet.size(), ' res eet visual: ', res_eet[:, :10])

def test04():
    torch.set_printoptions(precision=3, sci_mode=False)
    from transformers import BartTokenizer, BartModel
    torch.set_grad_enabled(False)

    tokenizer = BartTokenizer.from_pretrained('facebook/bart-base')
    ts_model = BartModel.from_pretrained('facebook/bart-base').cuda()
    from eet.transformers.modeling_bart import EETBartModel
    eet_model = EETBartModel.from_pretrained('facebook/bart-base', batch_size, seq_len)

    cfg = ts_model.config
    print(cfg)

    inputs = tokenizer("Hello, my dog is cute", return_tensors="pt")
    for k, v in inputs.items():
        print(k, v)
    # outputs = ts_model(**inputs)
    # for k, v in ts_model.state_dict().items():
    #     print(k, " size: ", v.size()) 

    input_ids = inputs['input_ids'].cuda()
    attention_mask = inputs['attention_mask'].cuda()

    for i in range(loop):
        res_eet = eet_model(input_ids=input_ids, encoder_out=None, first_pass=True)
    
    torch.cuda.synchronize()
    t1 = time.perf_counter()
    for i in range(loop):
        res_eet = eet_model(input_ids=input_ids, encoder_out=None, first_pass=True)
    torch.cuda.synchronize()
    t2 = time.perf_counter()
    time_eet = t2 - t1
    
    torch.cuda.synchronize()
    t3 = time.perf_counter()
    for i in range(loop):
        res_ts = ts_model(input_ids=input_ids, attention_mask=attention_mask)
    torch.cuda.synchronize()
    t4 = time.perf_counter()
    time_ts = t4 - t3

    # print("ts output: ", res_ts.last_hidden_state)
    # print("eet output: ", res_eet)
    print('Time for eet: ', time_eet)
    print('Time for Transformers: ', time_ts)
    print('SpeedUp is ', time_ts / time_eet)

def test05():
    model_id_or_path = "/root/data/transformer_file/transformer_model_pt/checkpoint_32_head_2.pt"
    model_dict = torch.load(model_id_or_path)
    for k, v in model_dict['model'].items():
        print(k)


if __name__ == "__main__":
    main()
