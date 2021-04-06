We tested the performance of EET on two GPU hardware platforms. We chose pytorch, NVIDIA Faster Transformers, and lightseq implementations for comparison.

### We show GPT2 inference performance here.

* RTX 2080ti

<div  align="left"> <img src="./image/gpt2_context_2080ti.jpg" width = "700" height = "299" alt="gpt2_context_2080ti"/></div>

<div  align="left"> <img src="./image/hidden_unit_2080ti.jpg" width = "700" height = "318" alt="hidden_unit_2080ti"/></div>

Medium sized model(hidden_units=1024,max_seq_len=768),compare with lightseq:
<div  align="left"> <img src="./image/1024model_lightseq.png" width = "700" height = "318" alt="1024model_lightseq"/></div>

Small-scale model(hidden_units=768,max_seq_len=128),compare with lightseq:
<div  align="left"> <img src="./image/768model_lightseq.png" width = "700" height = "318" alt="768model_lightseq"/></div>

* A100

<div  align="left"> <img src="./image/gpt2_context_A100.jpg" width = "700" height = "299" alt="gpt2_context_A100"/></div>

<div  align="left"> <img src="./image/hidden_unit_A100.jpg" width = "700" height = "318" alt="hidden_unit_A100"/></div>


### We show BERT inference performance here.

* RTX 2080ti

<div  align="left"> <img src="./image/bert_speedup_2080ti.jpg" width = "700" height = "315" alt="bert_speedup_2080ti"/></div>

* A100

<div  align="left"> <img src="./image/bert_speedup_A100.jpg" width = "700" height = "315" alt="bert_speedup_A100"/></div>
