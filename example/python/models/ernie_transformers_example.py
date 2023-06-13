import torch
import numpy as np
from transformers import ErnieModel
from eet.transformers.modeling_ernie import EETErnieModel
import time

using_half = True
seq_len = 64
batch = 4
loop = 100

def setup_seed(seed):
     torch.manual_seed(seed)
     torch.cuda.manual_seed_all(seed)
     np.random.seed(seed)
    #  random.seed(seed)
     torch.backends.cudnn.deterministic = True

def main():
    setup_seed(1)
    torch.set_printoptions(sci_mode=False)
    torch.set_grad_enabled(False)
    # 输入数据构造，实际业务输入应该是tokens
    input = np.random.randint(1000, 9000, seq_len * batch, dtype="int64")
    input_ids = torch.from_numpy(input).long().reshape(batch, seq_len).cuda()
    attention_mask = None
    task_type_ids = None

    data_type = torch.float32
    if using_half:
        data_type = torch.float16
    
    # load model,eet 需要传入最大batch_size和数据类型
    ts_model = ErnieModel.from_pretrained('nghuyong/ernie-1.0-base-zh').to(data_type).cuda()
    eet_model = EETErnieModel.from_pretrained('nghuyong/ernie-1.0-base-zh', max_batch=batch, max_seq_len=seq_len, data_type=data_type)

    params = sum(p.numel() for p in ts_model.parameters())
    print("params: ", params)

    # eet inference
    torch.cuda.synchronize()
    t1 = time.perf_counter()
    for i in range(loop):
        res_eet = eet_model(input_ids, attention_mask=attention_mask, task_type_ids=task_type_ids)
    torch.cuda.synchronize()

    t2 = time.perf_counter()
    time_eet = t2 - t1
    torch.cuda.synchronize()

    torch.cuda.synchronize()
    t3 = time.perf_counter()
    with torch.no_grad():
        for i in range(loop):
            res_ts = ts_model(input_ids, attention_mask=attention_mask, task_type_ids=task_type_ids)
    torch.cuda.synchronize()
    t4 = time.perf_counter()
    time_ts = t4 -t3
    
    print("res ts: ", res_ts.last_hidden_state, res_ts.last_hidden_state.shape)
    print("res eet: ", res_eet[0], res_eet[0].shape)
    error = res_eet[0] - res_ts.last_hidden_state
    print("error: ", error[:, :, :10])
    print('Time for EET : ', time_eet)
    print('Time for Transformers: ', time_ts)
    print('SpeedUp is ', time_ts / time_eet)

if __name__ == '__main__':
    main()
