import sys
import torch
from fairseq.data.dictionary import Dictionary
from eet import GPT2Config
args_config_map = {
    "activation_function":"activation_fn",
    "n_embd": "decoder_embed_dim",
    "n_head": "decoder_attention_heads",
    "n_layer": "decoder_layers",
    "n_positions": "max_target_positions",
    }

task_specific_params = {
    "text-generation": {
      "do_sample": True,
      "max_length": 50
    }
  }
architectures = [
      "TransformerDecoder"
    ]

dictionary = Dictionary.load('../../resource/dict.txt')
vocab_size = len(dictionary)

pretrained_model = torch.load('../../resource/checkpoint_best.pt')

if pretrained_model['args'] is not None:
  args = pretrained_model['args']
else:
  print('No args found, please add print and modify the code')
  sys.exit() 

gpt2_config = GPT2Config()
gpt2_config.vocab_size = vocab_size
gpt2_config.task_specific_params = task_specific_params
gpt2_config.architectures = architectures
gpt2_config.full_seqlen = 512 # 用来表示能支持的最长提示词长度，用于内部申请显存，尽量根据业务设置的小一点
for k,v in args_config_map.items():
    setattr(gpt2_config, k, getattr(args, v)) 

gpt2_config.save_pretrained('../../resource')

