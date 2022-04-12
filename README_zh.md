## Easy and Efficient Transformer

<div align='right' ><font size="1"><b><a href="./README.md">ENGLISH README</a></b> </font></div>


<div  align="center"> <img src="./doc/image/EETblueLOGO.png" width = "600" height = "180" alt="EET" align=center /></div>
</br>
EET（Easy But Efficient Transformer）是一款针对Transformer-based大模型和长序列场景的高性能pytorch推理插件。

## 功能特性
* 高性能：设计高度优化的CUDA内核，参考[NVIDIA Faster Transformer](https://github.com/NVIDIA/DeepLearningExamples/tree/master/FasterTransformer/v3.1)。 
* 灵活： 提供算子级和模型级API，允许用户自定义模型或者只更新部分算法逻辑。  
* 易于使用： EET可以直接集成到Fairseq和Transformes中，只需改动很少几行代码即可完成从训练到推理的转换。  
* 新增pipelines功能，提升用户体验，支持fairseq模型和transformers模型。
* bert模型整体性能加速1.2x到7.x倍，gpt模型整体性能加速2.x到7.x倍。


EET已经应用于多款网易线上服务，如逆水寒，网易云音乐，Lofter，天谕等。未来EET将致力于万亿模型的线上推理。

* [解码机制](#解码机制)
* [快速开始](#快速开始)
  * [环境](#环境)
  * [安装](#安装)
    * [源码安装](#源码安装)
    * [docker镜像安装](#docker镜像安装)
  * [运行](#运行)
    * [Model API](#model-api)
    * [pipelines方式](#pipelines方式)
* [支持模型](#支持模型)
* [使用方式](#使用方式)
* [性能](#性能)
* [TODO](#todo)
* [Cite Us](#cite-us)
* [联系我们](#联系我们)

| Frameworks |  maximum model size | maximum sequence length |Performance |Bert|GPT-2|Op-level|Fairseq support|Transformers support|dynamic batch & variable inputs|
|--------------------|-------------|------------------|------------|----|-----|--------|---------------|--------------------|-------------------------------|        
| EET                |16384       | 16384            |highest     | Y  |  Y  |    Y   |       Y       |          Y         |              Y                |
| Faster Transformer |  特定数字的倍数(128,256,384,512)        | 1024             |high        | Y  |  Y  |    N   |       N       |          N         |              N                |
| TensorRT           |  1024        | 1024             |high        | Y  |  N  |    N   |       N       |          N         |              N                | 
| LightSeq           |  1024        | 1024             |high        | Y  |  Y  |    N   |       N       |          N         |              Y                |  
| TurboTransformer   | 1024        | 1024             |medium      | Y  |  Y  |    N   |       N       |          Y         |              Y                | 
| ONNX               | non-limited | non-limited      |slow        | Y  |  Y  |    Y   |       N       |          N         |              Y                |  

## 解码机制
<div  align="left"> <img src="./doc/image/pre_decoding.svg" width = "700" height = "350" alt="bert"/></div>


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
我们提供两种运行方式：

#### Model API

用法非常简单，仅三行代码就可以完全适配transformers。需要注意的是，EET的GPT模型只支持左边打padding，其他模型支持右边打padding。

* 具体用法   
>1、 如何加载EET模块；如何加载预训练模型；如何做推理  
<div  align="left"> <img src="./doc/image/use_bert.png" width = "850" height = "325" alt="useofbert"/></div>

>2、如何构建model服务   
我们选择[service-streamer](https://github.com/ShannonAI/service-streamer)来创建我们的模型服务。

>3、如何自定义模型结构   
你可以参考下面列出来的API列表，只需要修改[python/eet](./python/eet)下的文件就可以很方便的构建自己的模型结构。

>4、如何将EET插入fairseq和transformers      
如果你想将EET插入fairseq或者transformers里，对照下面给出的类的对照表，替换修改即可，并且我们提供了非常丰富的model api，可自行组合实现，更方便的是我们提供了pipelines方式，几行代码即可实现不同任务的推理。

用户自行使用不同的model去做推理，具体使用方式见[example/python/models](./example/python/models/model)

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


## 使用方式
EET提供了python API接口([python/eet](./python/eet))

* API  
    1.模型API：我们提供了现成的GPT2和BERT模型API，因此您可以加载PyTorch模型并仅用几行代码来进行推理，就像使用Fairseq或Transformers一样。  
    <b>EET 和 fairseq 类对照表</b>
    | EET | fairseq| Remarks | 
    |-------|-------------|-------------| 
    | EETTransformerDecoder | TransformerDecoder | No |
    | EETTransformerDecoderLayer | TransformerDecoderLayer | No |
    | EETTransformerAttention | MultiheadAttention | No |
    | EETTransformerFeedforward | TransformerDecoderLayer | fusion of multiple small operators |
    | EETTransformerEmbedding | Embedding + PositionalEmbedding | No |
    | EETTransformerLayerNorm | nn.LayerNorm | No |

    <b>EET和transformers 类对照表</b>
    | EET | transformers| Remarks | 
    |---------------|-----------------|----| 
    | EETBertModel | BertModel | No |
    | EETBertEncoder | BertEncoder | No |
    | EETBertEncoderLayer | BertLayer | No |
    | EETBertAttention | BertAttention | No |
    | EETBertFeedforward | BertIntermediate + BertOutput | No |
    | EETBertEmbedding | BertEmbeddings | No |
    | EETGPT2Model | GPT2Model | No |
    | EETGPT2Decoder | GPT2Model | transformers has no GPT2Decoder |
    | EETGPT2DecoderLayer | Block | No |
    | EETGPT2Attention | Attention | No|
    | EETGPT2Feedforward | MLP | No |
    | EETGPT2Embedding | nn.Embedding | No |
    | EETLayerNorm | nn.LayerNorm | No |

    为了更好的适配transformers，我们根据transformers扩充了支持model api，譬如对于bert模型，我们新增了如下的api，用于支持不同的任务：

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

    其他模型结构均有做此类设计。

    2.算子API：我们提供了Transformer模型所需的所有算子，例如multiheadattention，layernorm，FFN等，您可以结合不同的算子来构建不同的模型结构。

    | operators APIs | Remarks | 
    |-------|-------------|
    | masked_multi_head_attention | GPT2 self_attention |
    | cross_multi_head_attention | cross_attention | 
    | multi_head_attention | Bert self_attention | 
    | ffn | FeedForwardNetwork |
    | embedding | transformers & fairseq |
    | layernorm | nn.LayerNorm |

## 性能
在不同场景下测试EET的性能数据如下：
Note : 在总时间的测试中，假设了上下文的比例为５０％
* A100 (batch_size=4, max_sequence_length=1024, context_length=512, precision=half)
  | Model Name | Params | Layers | Hidden_units | inference time of per-token | total time of 1024 tokens |
  |-------------|-------|--------|--------------|-----------------------------|---------------------------|
  | GPT-3 Small| 125M   | 12     | 768          | 2.69ms                         | 1.38s                  |
  | GPT-3 Medium | 350M | 24     | 1024         | 5.58ms                         | 2.86s                  |  
  | GPT-3 Large | 760M  | 24     | 1536         | 6.64ms                         | 3.41s                  |
  | GPT-3 XL   | 1.3B   | 24     | 2048         | 7.3m                           | 3.76s                  |
  | GPT-3 2.7B | 2.7B   | 32     | 2560         | 46.1ms                         | 23.6s                  |
  | GPT-3 6.7B | 6.7B   | 32     | 4096         | 17.2ms                         | 8.85s                  |
  | GPT-3 13B | 13B     | 40     | 5120         | 29.3ms                         | 15.12s                 |

* A100 (batch_size=16, max_sequence_length=1024, context_length=512, precision=half)
  | Model Name | Params | Layers | Hidden_units | inference time of per-token | total time of 1024 tokens |
  |-------------|-------|--------|--------------|-----------------------------|---------------------------|
  | GPT-3 Small| 125M   | 12     | 768          | 2.84ms                         | 1.46s                     |
  | GPT-3 Medium | 350M | 24     | 1024         | 6ms                         | 3.11s                    |  
  | GPT-3 Large | 760M  | 24     | 1536         | 7.39ms                         | 3.80s                    |
  | GPT-3 XL   | 1.3B   | 24     | 2048         | 8.27m                         |  4.26s                   |
  | GPT-3 2.7B | 2.7B   | 32     | 2560         | 116ms                        |  59.8s                      |
  | GPT-3 6.7B | 6.7B     | 32     | 4096         |  23.18ms                | 12.25s                |
  | GPT-3 13B | 13B     | 40     | 5120         | 43.42ms                 | 22.58s                |


### 推理性能

GPT及Bert模型推理详细性能数据请点击[链接](https://github.com/NetEase-FuXi/EET/blob/main/doc/benchmark.md)查看



## TODO
1. int8
2. sparse

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
您可以将您的问题发布在github issues。 
