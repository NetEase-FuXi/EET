import torch
import numpy as np
from torch.nn.parameter import Parameter
from eet.transformers.modeling_vit import EETViTModel
from eet.transformers.modeling_clip import EETCLIPModel
from transformers import CLIPModel, CLIPProcessor
from PIL import Image
import requests
import time

using_half = False
batch_size = 5
seq_len = 64
loop = 100

def main():
    torch.set_printoptions(precision=3, sci_mode=False)
    torch.set_grad_enabled(False)

    data_type = torch.float32

    ts_model = CLIPModel.from_pretrained("openai/clip-vit-base-patch32").cuda()
    processor = CLIPProcessor.from_pretrained("openai/clip-vit-base-patch32")
    if using_half:
        data_type = torch.float16
        ts_model = ts_model.half()
    
    # url = "http://images.cocodataset.org/val2017/000000039769.jpg"
    # image = Image.open("/root/3090_project/git/0406/EET/example/python/pipelines/images/cat.jpg")
    # inputs = processor(text=["a photo of a cat", "a photo of a dog"], images=image, return_tensors="pt", padding=True)
    # pixel_values = inputs['pixel_values'].to(data_type).cuda()
    # input_ids = inputs['input_ids'].cuda()
    # attention_mask = inputs['attention_mask'].to(data_type).cuda()
    
    input_ids = np.random.randint(1000, 9000, seq_len * batch_size, dtype="int64")
    input_ids = torch.from_numpy(input_ids).long().reshape(batch_size, seq_len).cuda()
    pixel_values = torch.from_numpy(np.random.random((batch_size, 3, 224, 224))).to(data_type).cuda()
    print(input_ids,input_ids.size(),pixel_values,pixel_values.size())

    for i in range(loop):
        res_ts = ts_model(input_ids, pixel_values, attention_mask=None)

    t1 = time.perf_counter()
    with torch.no_grad():
        for i in range(loop):
            res_ts = ts_model(input_ids, pixel_values, attention_mask=None)
    torch.cuda.synchronize()
    t2 = time.perf_counter()
    time_ts = t2 - t1

    num_channels = pixel_values.shape[1]
    eet_model = EETCLIPModel.from_pretrained("openai/clip-vit-base-patch32", max_batch=batch_size, num_channels=num_channels, data_type=data_type)

    t3 = time.perf_counter()
    for i in range(loop):
        res_eet = eet_model(input_ids, pixel_values, attention_mask=None)
    torch.cuda.synchronize()
    t4 = time.perf_counter()
    time_eet = t4 - t3

    print('Time for EET : ', time_eet)
    print('Time for Transformers: ', time_ts)
    print('SpeedUp is ', time_ts / time_eet)

    print("ts output: ", res_ts.logits_per_image,res_ts.logits_per_image.size())
    print("ts output: ", res_ts.logits_per_text)
    print("ts similarity score: ", res_ts.logits_per_image.softmax(dim=1))
    print("eet output: ", res_eet)
    print("eet similarity score: ",res_eet[0].size(), res_eet[0].softmax(dim=1))

if __name__ == "__main__":
    main()
