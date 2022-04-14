## Easy and Efficient Transformer 
<div align='right' ><font size="1"><b><a href="./README_zh.md">中文README</a></b> </font></div>


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

EET(Easy and Efficient Transformer) is an efficient Pytorch inference plugin focus on Transformer-based models with large model sizes and long sequences.

## Features

- 1、High performance.Design highly optimized CUDA kernels.  
- 2、Flexible.Provide op-level、model-level APIs and pipelines for different needs.  
- 3、Easy to use.A few lines of code will do the trick.
- 4、embed_dim up to 16384 and sequence lengths up to 4096.  
- 5、Adaptation to mainstream frameworks,include transformers and fairseq.  

----

* [Supported Models](#supported-models)
* [Quick Start](#quick-start)
  * [Environment](#environment)
  * [Installation](#installation)
    * [From Source](#from-source)
    * [From Docker](#from-docker)
  * [Run](#run)
    * [operators API](#operators-api)
    * [Model API](#model-api)
    * [pipelines](#pipelines)
* [Performance](#performance)
* [Cite Us](#cite-us)
* [Contact us](#contact-us)


## Supported Models

| Model | Since version | 
|-------|-------------|
| GPT2 | 0.0.1 beta |
| Bert | 0.0.1 beta | 
| Roberta | 1.0 | 
| Albert | 1.0 |
| Vit | 1.0 |
| Clip | 1.0 |
| Distilbert | 1.0 |

## Quick Start

### Environment

* cuda:>=10.1 
* python:>=3.7 
* gcc:>= 7.4.0 
* torch:>=1.5.0 
* numpy:>=1.19.1 
* fairseq
* transformers

The above environment is the minimum configuration, and it is best to use a newer version.

### Installation

Recommend using docker images

#### From Source
If you are installing from source, you will need install the necessary [environment](#environment).Then, proceed as follows: 

```bash
$ git clone https://github.com/NetEase-FuXi/EET.git
$ pip install .
```
Recommend using nvcr.io/nvidia/pytorch:21.12-py3 and other series of images, you can also use the provided Dockerfile file

#### From Docker

```bash
$ git clone https://github.com/NetEase-FuXi/EET.git
$ docker build -t eet_docker:0.1 .
$ nvidia-docker run -it --net=host -v /your/project/directory/:/root/workspace  eet_docker:0.1 bash
```
the EET and its required environment are installed in docker.

### Run

We offer three types of operation.

#### operators API

We provide all the operators required for Transformer models. You can combine different kernels to build different model structures.
- operators API table

    | operators API | Remarks | 
    |-------|-------------|
    | masked_multi_head_attention | GPT2 self_attention |
    | cross_multi_head_attention | cross_attention | 
    | multi_head_attention | Bert self_attention | 
    | ffn | FeedForwardNetwork |
    | embedding | transformers & fairseq |
    | layernorm | nn.LayerNorm |

- how to use

    You can refer to Operators APIs listed above to build your own model structure, just by modifying the files under [python/eet](./python/eet).

#### Model API
EET provides python User-friendly API([python/eet](./python/eet)), integrated into Fairseq and Transformers with just a few lines of code. It should be noted that for gpt we only support padding on the left.
    
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

In order to better fit transformers, we have expanded the support model api based on transformers,For example, for the bert model, we have added the following api to support different tasks:      

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

How to inference 

<div  align="left"> <img src="./doc/image/use_bert.png" width = "850" height = "325" alt="useofbert"/></div>

Please refer to [example/python/models](example/python/models/).

#### pipelines

EET provides a ready-made pipelines approach to provide pipeline usage options for different tasks based on the different model structures supported by EET.

The usage is very simple:

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

support task：

| Task | Since version | 
|-------|-------------|
| text-classification | 1.0 |
| token-classification | 1.0 | 
| question-answering | 1.0 | 
| fill-mask | 1.0 |
| text-generation | 1.0 |
| image-classification | 1.0 |
| zero_shot_image_classification | 1.0 |

Later on, as more and more models are supported by EET, more and more pipeline tasks will be supported.

[example/python/pipelines](./example/python/pipelines),In these sample task codes, we also provide model api examples to implement the same tasks.


## Performance

Detailed performance data of GPT and Bert model inference can be viewed at [link](https://github.com/NetEase-FuXi/EET/blob/main/doc/benchmark.md)
* gpt-A100

<div  align="left"> <img src="./doc/image/a100_prompt.png" width = "700" height = "387" alt="a100_prompt"/></div>

* bert-2080ti
<div  align="left"> <img src="./doc/image/bert_ft.png" width = "700" height = "386" alt="bert_ft"/></div>

## Cite Us

If you use EET in your research, please cite the following paper.We also have a share on ZhiYuan LIVE, share link: https://event.baai.ac.cn/activities/325

```
@misc{https://doi.org/10.48550/arxiv.2104.12470,
  doi = {10.48550/ARXIV.2104.12470},
  url = {https://arxiv.org/abs/2104.12470},
  author = {Li, Gongzheng and Xi, Yadong and Ding, Jingzhen and Wang, Duan and Liu, Bai and Fan, Changjie and Mao, Xiaoxi and Zhao, Zeng},
  keywords = {Computation and Language (cs.CL), FOS: Computer and information sciences, FOS: Computer and information sciences},
  title = {Easy and Efficient Transformer : Scalable Inference Solution For large NLP model},
```

## Contact us
You can post your problem with github issues.And you can contact us by email.

ligongzheng@corp.netease.com, dingjingzhen@corp.netease.com ,zhaosida@corp.netease.com

And if you are interested in high performance computing, deep learning, you can join us, we welcome you. Just send your resume directly to the above email address.

