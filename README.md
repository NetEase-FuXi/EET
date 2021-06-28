## Easy and Efficient Transformer 
<div align='right' ><font size="1"><b><a href="./README_zh.md">中文README</a></b> </font></div>


<div  align="center"> <img src="./doc/image/EETblueLOGO.png" width = "600" height = "180" alt="EET" align=center /></div>
</br>
EET(Easy and Efficient Transformer) is an efficient Pytorch inference plugin focus on Transformer-based models with large model sizes and long sequences.

## Features

>1、Pre-padding decoding. Pre-padding keep the relative position embedding remain unchanged within the context and the generated sequence, reducing the gap between training and inference. Basic on this, we achieve parallel inference for the context and incremental decoding for generated token sequence.   
>2、High performance.  Design highly optimized CUDA kernels, referencing to NVIDIA [Faster Transformer](https://github.com/NVIDIA/DeepLearningExamples/tree/master/FasterTransformer/v3.1) with advanced optimization.  
>3、Flexible.  Provide op-level and model-level APIs, allowing users to construct their model or upgrade partial algorithm flexible.  
>4、Easy to use. EET could be integrated into Fairseq and Transformes directly by replacement of sepcified files, without any code change.  
>5、Dynamic batch. EET supports dynamic batch, which changes the order of the batch according to the reorder_state and can end a batch early.    
>6、Extra-large dimension and extra-long sequences. EET supports GPT hidden_units up to 16384 and sequence lengths up to 4096.  


EET has been applied to a variety of NetEase online services,such as NiShuiHan, NetEase's cloud music, TianYu, Lofter, etc. In the future, EET will work on urtra-large-scale model inference of trillion parameters.   

* [Decoding mechanism](#decoding-mechanism)
* [Quick Start](#quick-start)
  * [Environment](#environment)
  * [Installation](#installation)
    * [From Source](#from-source)
    * [From Docker](#from-docker)
  * [Run](#run)
    * [run BERT in Transformers](#run-bert-in-transformers)
    * [run GPT2 in Transformers](#run-gpt2-in-transformers)
    * [run GPT2 in Fairseq](#run-gpt2-in-fairseq)
* [Supported Models](#supported-models)
  * [GPT2](#gpt2)
  * [BERT](#bert)
* [Usage](#usage)
* [APIs](#apis)
* [Performance](#performance)
  * [We show GPT2 inference performance here\.](#we-show-gpt2-inference-performance-here)
  * [We show BERT inference performance here\.](#we-show-bert-inference-performance-here)
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

### Installation
#### From Source
If you are installing from source, you will need install the necessary [environment](#environment).Then, proceed as follows: 

```bash
$ git clone git@github.com:NetEase-FuXi/EET.git
$ pip install transformers==3.5.0
$ pip install fairseq==0.10.0
$ pip install .
```
Due to the compilation of a large number of cuda kernels, the installation time is relatively long, please be patient. 

#### From Docker

```bash
$ git clone git@github.com:NetEase-FuXi/EET.git
$ cd EET/docker
$ docker build -t your_docker_name:your_docker_version .
$ nvidia-docker run -it --net=host -v /your/project/directory/:/root/workspace  your_Docker_Name:your_docker_version bash
```
EET has been installed in the docker.

### Run
#### run BERT in Transformers
```bash
$ cd EET/example  
$ python bert_transformers_example.py
```

#### run GPT2 in Transformers
```bash
$ cd EET/example    
$ python gpt2_transformers_example.py
```

#### run GPT2 in Fairseq
```bash
$ cd EET
$ wget https://github.com/NetEase-FuXi/EET/releases/download/EET_V0.0.1_fairseq0.10.0_transformers3.5.0/resource.zip
$ cd example 
$ python gpt2_fairseq_example.py
```

## Supported Models

We currenly support the GPT-2, Bert.

### GPT2

<div  align="left"> <img src="./doc/image/gpt2.jpg" width = "400" height = "632" alt="gpt2"/></div>

### BERT

<div  align="left"> <img src="./doc/image/bert.jpg" width = "400" height = "463" alt="bert"/></div>

## Usage
EET provides python User-friendly APIs([python/eet](./python/eet)), integrated into Fairseq and Transformers with just a few lines of code. 

>1、How to inference 
<div  align="left"> <img src="./doc/image/use_bert.png" width = "850" height = "325" alt="useofbert"/></div>

>2、How to customize model  
You can refer to Operators APIs listed below to build your own model structure, just by modifying the files under [python/eet](./python/eet).

>3、How to integrate EET into fairseq  
Replace the original transformer.py in Fairseq with our transformer.py and reinstall the Fairseq, that is all !
[Transformer.py](./python/eet/fairseq/transformer.py) in EET corresponds to the fusion of [transformer.py](https://github.com/pytorch/fairseq/blob/master/fairseq/models/transformer.py) and [transformer_layer.py](https://github.com/pytorch/fairseq/blob/master/fairseq/modules/transformer_layer.py) in fairseq.

>4、How to integrate EET into Transformers  
Replace the original modeling_bert.py and odeling_gpt2.py in Transformers with our modeling_bert.py and modeling_gpt2.py and reinstall the Transformers, that is all !
[modeling_bert.py](./python/eet/transformers/modeling_bert.py) in EET corresponds to [modeling_bert.py](https://github.com/huggingface/transformers/blob/v3.0.2/src/transformers/modeling_bert.py) in transformers;[modeling_gpt2.py](./python/eet/transformers/modeling_gpt2.py) in EET corresponds to [modelling_gpt2.py](https://github.com/huggingface/transformers/blob/v3.0.2/src/transformers/modelling_gpt2.py) in transformers.

>5、How to make a server  
We choose  [service-streamer](https://github.com/ShannonAI/service-streamer) to make the model server, building the service based on your python project directly. 
Please make sure the dynamic-batch is open if you want a higher throughput.

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
* 3090 (batch_size=4, max_sequence_length=1024, context_length=512, precision=half)
    | Model Name | Params | Layers | Hidden_units | inference time of per-token | total time of 1024 tokens |
    |-------------|-------|--------|--------------|-----------------------------|---------------------------|
    | GPT-3 Small| 125M   | 12     | 768          | 3ms                         | 1.67s                     |
    | GPT-3 Medium | 350M | 24     | 1024         | 7ms                         | 3.546s                    |  
    | GPT-3 Large | 760M  | 24     | 1536         | 8ms                         | 4.361s                    | 
    | GPT-3 XL   | 1.3B   | 24     | 2048         | 10m                         |  5.091s                   | 
    | GPT-3 2.7B | 2.7B   | 32     | 2560         | 60ms                        |  31s                      | 
    | GPT-3 5B | 5B       | 45     | 3072         | 25ms                        |  13.149s                  |
    | GPT-3 8B   | 8B     | 40     | 4096         |  30ms                   | 15.97s                        |
    | GPT-3 10B | 10B     | 36     | 5120         | outOfMemory             | outOfMemory                   |

* 3090 (batch_size=16, max_sequence_length=1024, context_length=512, precision=half)
  | Model Name | Params | Layers | Hidden_units | inference time of per-token | total time of 1024 tokens |
  |-------------|-------|--------|--------------|-----------------------------|---------------------------|
  | GPT-3 Small| 125M   | 12     | 768          | 3ms                         | 1.61s                     |
  | GPT-3 Medium | 350M | 24     | 1024         | 6ms                         | 3.416s                    |  
  | GPT-3 Large | 760M  | 24     | 1536         | 8ms                         | 4.402s                    |
  | GPT-3 XL   | 1.3B   | 24     | 2048         | 11m                         |  6.374s                   | 
  | GPT-3 2.7B | 2.7B   | 32     | 2560         | 175ms                        |  91s                      |
  | GPT-3 5B | 5B       | 45     | 3072         | 31ms                        |  19.565s                   |
  | GPT-3 8B   | 8B     | 40     | 4096         |  outOfMemory                | outOfMemory                |
  | GPT-3 10B | 10B     | 36     | 5120         | outOfMemory                 | outOfMemory                |
  
We tested the performance of EET on two GPU hardware platforms. We chose pytorch, NVIDIA Faster Transformers, and lightseq implementations for comparison.

### We show GPT2 inference performance here.

* RTX 2080ti (batch_size=4, hidden_units=1024, sequence_length=1024, precision=fp16)

<div  align="left"> <img src="./doc/image/2080_gpt.svg" width = "700" height = "299" alt="gpt2_context_2080ti"/></div>

* RTX 2080ti (batch_size=4, context_ratio=50%, sequence_length=1024, precision=fp16)
<div  align="left"> <img src="./doc/image/gpt1.svg" width = "700" height = "318" alt="hidden_unit_2080ti"/></div>

* A100 (batch_size=4, hidden_units=1024, sequence_length=1024, precision=fp16)

<div  align="left"> <img src="./doc/image/a100_gpt.svg" width = "700" height = "299" alt="gpt2_context_A100"/></div>

* A100 (batch_size=4, context_ratio=50%, sequence_length=1024, precision=fp16)

<div  align="left"> <img src="./doc/image/gpt2.svg" width = "700" height = "318" alt="hidden_unit_A100"/></div>

Medium size model(hidden_units=1024,max_seq_len=768),compare with lightseq:
<div  align="left"> <img src="./doc/image/lightseq1.svg" width = "700" height = "318" alt="1024model_lightseq"/></div>

Small size model(hidden_units=768,max_seq_len=128),compare with lightseq:
<div  align="left"> <img src="./doc/image/lightseq2.svg" width = "700" height = "318" alt="768model_lightseq"/></div>



### We show BERT inference performance here.

* RTX 2080ti

<div  align="left"> <img src="./doc/image/bert_2080.svg" width = "700" height = "315" alt="bert_speedup_2080ti"/></div>

* A100

<div  align="left"> <img src="./doc/image/bert_a100.svg" width = "700" height = "315" alt="bert_speedup_A100"/></div>

## TODO
1. int8
2. sparse


## Contact us
You can post your problem with github issues.
