## Easy and Efficient Transformer

<div align='right' ><font size="1"><b><a href="./README.md">ENGLISH README</a></b> </font></div>


<div  align="center"> <img src="./doc/image/EETblueLOGO.png" width = "600" height = "180" alt="EET" align=center /></div>
</br>

<p align="center">
    <a href="https://github.com/NetEase-FuXi/EET/blob/main/LICENSE">
        <img alt="GitHub license" src="./doc/image/license.svg">
    </a>
    <a href="https://github.com/NetEase-FuXi/EET/tree/main/example/python">
        <img alt="GitHub release" src="./doc/image/example.svg">
    </a>
    <a href="https://github.com/NetEase-FuXi/EET/releases">
        <img alt="release" src="./doc/image/release.svg">
    </a>
</p>

EET（Easy But Efficient Transformer）是一款针对Transformer-based大模型和长序列场景的高性能pytorch推理插件。

## 功能特性
* **新特性**🔥: 现已支持Baichuan、LLaMA等LLMs。
* 高性能：设计高度优化的CUDA内核。
* 灵活：提供了包括op api、model api和pipelines应对不同的需求 
* 易于使用： 几行代码即可完成。 
* 适配主流ai框架，包括fairseq和transformers。
* bert模型整体性能加速1.2x到7.x倍，gpt模型整体性能加速2.x到7.x倍。

---

- [Easy and Efficient Transformer](#easy-and-efficient-transformer)
- [功能特性](#功能特性)
- [支持模型](#支持模型)
- [快速开始](#快速开始)
  - [环境](#环境)
  - [安装](#安装)
    - [源码安装](#源码安装)
    - [docker镜像安装](#docker镜像安装)
  - [运行](#运行)
    - [operators API](#operators-api)
    - [Model API](#model-api)
    - [pipelines方式](#pipelines方式)
- [性能](#性能)
- [Cite Us](#cite-us)
- [联系我们](#联系我们)

## 支持模型

| Model | Since version | 
|-------|-------------|
| GPT2 | 0.0.1 beta |
| Bert | 0.0.1 beta | 
| Roberta | 1.0 | 
| Albert | 1.0 |
| Vit | 1.0 |
| Clip | 1.0 |
| Distilbert | 1.0 |
| Baichuan | 2.0 |
| LLaMA | 2.0 |

## 快速开始

### 环境

* cuda:>=10.1 
* python:>=3.7 
* gcc:>= 7.4.0 
* torch:>=1.5.0 
* numpy:>=1.19.1 
* fairseq
* transformers

上述环境是最低配置，最好是使用较新的版本。

推荐使用nvcr.io/nvidia/镜像

### 安装

推荐使用docker安装

#### 源码安装
如果从源代码安装，则需要安装必要的[environment](#environment)。然后，按以下步骤进行。 
```bash
$ git clone https://github.com/NetEase-FuXi/EET.git
$ pip install .
```

#### docker镜像安装

推荐使用nvcr.io/nvidia/pytorch:21.12-py3等系列镜像，也可使用提供的Dockerfile文件

```bash
$ git clone https://github.com/NetEase-FuXi/EET.git
$ docker build -t eet_docker:0.1 .
$ nvidia-docker run -it --net=host -v /your/project/directory/:/root/workspace  eet_docker:0.1 bash
```

此时，EET及其所需的环境均已安装在docker中。

### 运行
我们提供三种运行方式：

#### operators API

我们提供了Transformer模型所需的所有算子，例如multiheadattention，layernorm，FFN等，您可以结合不同的算子来构建不同的模型结构
- operators API 表

    | operators API | Remarks | 
    |-------|-------------|
    | masked_multi_head_attention | GPT2 self_attention |
    | cross_multi_head_attention | cross_attention | 
    | multi_head_attention | Bert self_attention | 
    | ffn | FeedForwardNetwork |
    | embedding | transformers & fairseq |
    | layernorm | nn.LayerNorm |

- 使用方式

    你可以参考上面列出的op API来构造自己的模型结构，只要修改[python/eet](./python/eet)下的文件就可以了。


#### Model API

EET提供了非常友好的model级api([python/eet](./python/eet))，适配fairseq和transformers模型只需要几行代码。
用法非常简单，仅三行代码就可以完全适配transformers。需要注意的是，EET的GPT模型只支持左边打padding，其他模型支持右边打padding。
    
<b>EET and fairseq class comparison table</b>
| EET | fairseq| Remarks | 
|-------|-------------|-------------| 
| EETTransformerDecoder | TransformerDecoder |  |
| EETTransformerDecoderLayer | TransformerDecoderLayer |  |
| EETTransformerAttention | MultiheadAttention |  |
| EETTransformerFeedforward | TransformerDecoderLayer | fusion of multiple small operators |
| EETTransformerEmbedding | Embedding + PositionalEmbedding |  |
| EETTransformerLayerNorm | nn.LayerNorm |  |

<b>EET and transformers class comparison table</b>
| EET | transformers| Remarks | 
|---------------|-----------------|----| 
| EETBertModel | BertModel |  |
| EETBertEncoder | BertEncoder |  |
| EETBertEncoderLayer | BertLayer |  |
| EETBertAttention | BertAttention |  |
| EETBertFeedforward | BertIntermediate + BertOutput |  |
| EETBertEmbedding | BertEmbeddings |  |
| EETGPT2Model | GPT2Model |  |
| EETGPT2Decoder | GPT2Model | transformers has no GPT2Decoder |
| EETGPT2DecoderLayer | Block |  |
| EETGPT2Attention | Attention | |
| EETGPT2Feedforward | MLP |  |
| EETGPT2Embedding | nn.Embedding |  |
| EETLayerNorm | nn.LayerNorm |  |

为了更好地适配transformers，我们在transformers的基础上扩展了支持模型的api,例如，对于bert模型，我们增加了以下api来支持不同的任务。     

| EET | transformers| Remarks | 
|---------------|-----------------|----| 
| EETBertForPreTraining | BertForPreTraining | No |
| EETBertLMHeadModel | BertLMHeadModel | No |
| EETBertForMaskedLM | BertForMaskedLM | No |
| EETBertForNextSentencePrediction | BertForNextSentencePrediction | No |
| EETBertForSequenceClassification | BertForSequenceClassification | No |
| EETBertForMultipleChoice | BertForMultipleChoice | No |
| EETBertForTokenClassification | BertForTokenClassification | No |
| EETBertForQuestionAnswering | BertForQuestionAnswering | No |

其他模型也都提供了相关api。

推理方法：

<div  align="left"> <img src="./doc/image/use_bert.png" width = "850" height = "325" alt="useofbert"/></div>

你也可以直接使用这些model api实现自己的特定任务，下面以fill-mask任务为例：

```python
from eet import EETRobertaForMaskedLM
from transformers import RobertaTokenizer
input = ["My <mask> is Sarah and I live in London"]
tokenizer = RobertaTokenizer.from_pretrained('roberta-base')
eet_roberta_model = EETRobertaForMaskedLM.from_pretrained(model_path,max_batch = max_batch_size,data_type = data_type)
# first step: tokenize
model_inputs = tokenizer(input,return_tensors = 'pt')
masked_index = torch.nonzero(model_inputs['input_ids'][0] == tokenizer.mask_token_id, as_tuple=False).squeeze(-1)
# second step: predict
prediction_scores = eet_roberta_model(model_inputs['input_ids'].cuda(),attention_mask = model_inputs['attention_mask'])
# third step: argmax
predicted_index = torch.argmax(prediction_scores.logits[0, masked_index]).item()
predicted_token = tokenizer.convert_ids_to_tokens(predicted_index)
```

请参考 [example/python/models](example/python/models/).

#### pipelines方式

EET提供了现成的pipelines方式，基于EET支持的不同模型结构，提供不同任务的pipeline使用方案。

使用方式非常简单：

```python
import torch
from eet import pipeline
max_batch_size = 1
model_path = 'roberta-base'
data_type = torch.float16
input = ["My <mask> is Sarah and I live in London"]
nlp = pipeline("fill-mask",model = model_path,data_type = data_type,max_batch_size = max_batch_size)
out = nlp(input)
```

具体支持：

| Task | Since version | 
|-------|-------------|
| text-classification | 1.0 |
| token-classification | 1.0 | 
| question-answering | 1.0 | 
| fill-mask | 1.0 |
| text-generation | 1.0 |
| image-classification | 1.0 |
| zero_shot_image_classification | 1.0 |

后续随着EET支持的模型越来越多，支持的pipeline任务也将越来越多。

使用方式见[example/python/pipelines](./example/python/pipelines),在这些任务示例代码中，我们也提供了model api示例来实现同样的任务。


## 性能

详细性能数据请点击[链接](https://github.com/NetEase-FuXi/EET/blob/main/doc/benchmark.md)查看
* gpt-A100

<div  align="left"> <img src="./doc/image/a100_prompt.png" width = "700" height = "387" alt="a100_prompt"/></div>

* bert-2080ti
<div  align="left"> <img src="./doc/image/bert_ft.png" width = "700" height = "386" alt="bert_ft"/></div>

## Cite Us

如果你在研究中使用EET，请引用以下论文，我们也有在智源LIVE上做分享，分享链接：https://event.baai.ac.cn/activities/325

```
@article{eet2022,
  title={Easy and Efficient Transformer : Scalable Inference Solution For large NLP model},
  author={Gongzheng Li, Yadong Xi, Jingzhen Ding, Duan Wang, Bai Liu, Changjie Fan, Xiaoxi Mao, Zeng Zhao},
  journal={	arXiv:2104.12470},
  year={2022}
}
```
## 联系我们
您可以将您的问题发布在github issues，也可以通过邮件与我们联系。

zhaosida@corp.netease.com, zhuangzhong@corp.netease.com, hzzhaozeng@corp.netease.com

如果你对高性能计算、深度学习感兴趣，欢迎加入我们团队，简历可发送上述邮箱。
