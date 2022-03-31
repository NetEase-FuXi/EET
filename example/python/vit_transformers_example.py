import torch
import numpy as np
from torch.nn.parameter import Parameter
from eet.transformers.modeling_vit import EETViTModel
from transformers import ViTFeatureExtractor, ViTModel
from PIL import Image
import requests
import time

using_half = True
batch_size = 1
loop = 100


def main():
    torch.set_grad_enabled(False)

    # image input
    # url = 'http://images.cocodataset.org/val2017/000000039769.jpg'
    # image = Image.open(requests.get(url, stream=True).raw)
    # feature_extractor = ViTFeatureExtractor.from_pretrained('google/vit-base-patch16-224-in21k')
    # inputs = feature_extractor(images=image, return_tensors="pt")

    # build pretrained model
    data_type = torch.float32
    ts_model = ViTModel.from_pretrained('google/vit-base-patch16-224-in21k').cuda()
    if using_half:
        ts_model = ts_model.half()
        data_type = torch.float16
    eet_model = EETViTModel.from_pretrained(
        'google/vit-base-patch16-224-in21k', max_batch=batch_size, data_type=data_type)
    
    # dummy input
    dummy_input = torch.from_numpy(np.random.random((batch_size, 3, 224, 224))).to(data_type).cuda()
    input_states = dummy_input
    attention_mask = None

    # Inference using transformers
    # The first inference takes a long time
    for i in range(loop):
        res_ts = ts_model(input_states, attention_mask)


    t3 = time.perf_counter()
    with torch.no_grad():
        for i in range(loop):
            res_ts = ts_model(input_states, attention_mask)
    t4 = time.perf_counter()
    time_ts = t4 - t3
    
    # Inference using EET
    t1 = time.perf_counter()
    for i in range(loop):
        res_eet = eet_model(input_states, attention_mask=attention_mask)
    t2 = time.perf_counter()
    time_eet = t2 - t1

    print('ts output: ', res_ts.pooler_output)
    print('eet output: ', res_eet)
    print('Time for EET: ', time_eet)
    print('Time for Transformers: ', time_ts)
    print('SpeedUp is ', time_ts / time_eet)


if __name__ == '__main__':
    main()