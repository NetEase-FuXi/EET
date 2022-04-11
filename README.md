## Easy and Efficient Transformer 
<div align='right' ><font size="1"><b><a href="./README_zh.md">中文README</a></b> </font></div>


<div  align="center"> <img src="./doc/image/EETblueLOGO.png" width = "600" height = "180" alt="EET" align=center /></div>
</br>
EET(Easy and Efficient Transformer) is an efficient Pytorch inference plugin focus on Transformer-based models with large model sizes and long sequences.

## Features

>1、Pre-padding decoding. Pre-padding keep the relative position embedding remain unchanged within the context and the generated sequence, reducing the gap between training and inference. Basic on this, we achieve parallel inference for the context and incremental decoding for generated token sequence.   
>2、High performance.  Design highly optimized CUDA kernels, referencing to NVIDIA [Faster Transformer](https://github.com/NVIDIA/DeepLearningExamples/tree/master/FasterTransformer/v3.1) with advanced optimization.  
>3、Flexible.  Provide op-level and model-level APIs, allowing users to construct their model or upgrade partial algorithm flexible.  
>4、Easy to use. EET could be integrated into Fairseq and Transformers directly by replacement of sepcified files, without any code change.  
>5、Dynamic batch. EET supports dynamic batch, which changes the order of the batch according to the reorder_state and can end a batch early.    
>6、Extra-large dimension and extra-long sequences. EET supports GPT hidden_units up to 16384 and sequence lengths up to 4096.  
>7、Support multiple models, including gpt2, bert, albert, roberta, vit.
>7、Support pipelines function to improve user experience, support fairseq model and transformers model
>9、The overall performance of the bert model is accelerated by 1.2x to 7.x times, and the overall performance of the gpt model is accelerated by 2.x to 7.x times.

EET has been applied to a variety of NetEase online services,such as NiShuiHan, NetEase's cloud music, TianYu, Lofter, etc. In the future, EET will work on urtra-large-scale model inference of trillion parameters.   

* [Decoding mechanism](#decoding-mechanism)
* [Quick Start](#quick-start)
  * [Environment](#environment)
  * [Installation](#installation)
    * [From Source](#from-source)
    * [From Docker](#from-docker)
  * [Run](#run)
    * [Model API](#model-api)
    * [pipelines](#pipelines)
* [Supported Models](#supported-models)
* [Usage](#usage)
* [APIs](#apis)
* [Performance](#performance)
* [TODO](#todo)
* [Contact us](#contact-us)

| Frameworks | maximum model size | maximum sequence length |Performance |Bert|GPT-2|Op-level|Fairseq support|Transformers support|dynamic batch & variable inputs|
|--------------------|-------------|------------------|------------|----|-----|--------|---------------|--------------------|-------------------------------|        
| EET                | 16384       | 16384            |highest     | Y  |  Y  |    Y   |       Y       |          Y         |              Y                |
| Faster Transformer | Multiples of specific numbers, such as 128, 256, 384, 512     | 1024             |high        | Y  |  Y  |    N   |       N       |          N         |              N                |
| TensorRT           | 1024        | 1024             |high        | Y  |  N  |    N   |       N       |          N         |              N                | 
| LightSeq           | 1024        | 1024             |high        | Y  |  Y  |    N   |       N       |          N         |              Y                |  
| TurboTransformer   | 1024        | 1024             |medium      | Y  |  Y  |    N   |       N       |          Y         |              Y                | 
| ONNX               | non-limited | non-limited      |slow        | Y  |  Y  |    Y   |       N       |          N         |              Y                |  

##  Decoding mechanism

<div  align="left"> <img src="./doc/image/pre_decoding.svg" width = "700" height = "350" alt="bert"/></div>


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

We offer two types of operation.

#### Model API

EET provides python User-friendly APIs([python/eet](./python/eet)), integrated into Fairseq and Transformers with just a few lines of code. It should be noted that for gpt we only support padding on the left.

>1、How to inference 
<div  align="left"> <img src="./doc/image/use_bert.png" width = "850" height = "325" alt="useofbert"/></div>

>2、How to customize model  
You can refer to Operators APIs listed below to build your own model structure, just by modifying the files under [python/eet](./python/eet).

>3、How to integrate EET into fairseq  and transformers 
If you want to insert EET into fairseq or transformers, you can replace it with the class cross-reference table given below, and we provide a very rich model api that can be combined to achieve it by yourself.

>4、How to make a server  
We choose  [service-streamer](https://github.com/ShannonAI/service-streamer) to make the model server, building the service based on your python project directly. 
Please make sure the dynamic-batch is open if you want a higher throughput.

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

support：

1、text-classification 

2、token-classification

3、question-answering 

4、fill-mask

5、text-generation

6、image-classification

7、zero_shot_image_classification

Later on, as more and more models are supported by EET, more and more pipeline tasks will be supported.

[example/python/pipelines](./example/python/pipelines),In these sample task codes, we also provide model api examples to implement the same tasks.


## Supported Models

We currenly support the GPT-2, Bert、Roberta、albert、clip、vit、distilbert.

## Usage

## APIs
1. model APIs:We provide ready-made APIs for GPT2 and Bert models.
    
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

 In order to better fit transformers, we have expanded the support model api based on transformers,For example, for the bert model, we have added the following api to support different tasks.
 
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

2. operators APIs:We provide all the operators required for Transformer models. You can combine different kernels to build different model structures
    | operators APIs | Remarks | 
    |-------|-------------|
    | masked_multi_head_attention | GPT2 self_attention |
    | cross_multi_head_attention | cross_attention | 
    | multi_head_attention | Bert self_attention | 
    | ffn | FeedForwardNetwork |
    | embedding | transformers & fairseq |
    | layernorm | nn.LayerNorm |

## Performance

### GPT-3 memory usage and performance
We measure the inference time and memory occupancy in different scenarios. 
Note : Total time are measured with 50% of the context
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
  

### Inference performance

Detailed performance data of GPT and Bert model inference can be viewed at [link](https://github.com/NetEase-FuXi/EET/blob/main/doc/benchmark.md)


## TODO
1. int8
2. sparse


## Contact us
You can post your problem with github issues.
