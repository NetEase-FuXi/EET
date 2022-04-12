We tested the performance of EET on two GPU hardware platforms. We chose pytorch, NVIDIA Faster Transformers, and lightseq implementations for comparison.

## We show GPT2 inference performance here.

* RTX 2080ti

<div  align="left"> <img src="./image/2080ti_prompt.png" width = "700" height = "386" alt="2080ti_prompt"/></div>

* A100

<div  align="left"> <img src="./image/a100_prompt.png" width = "700" height = "387" alt="a100_prompt"/></div>


* Compare with lightseq:
<div  align="left"> <img src="./image/lightseq.png" width = "700" height = "385" alt="lightseq"/></div>


* Compare with fastertransformer:
<div  align="left"> <img src="./image/gpt2_ft.png" width = "700" height = "386" alt="gpt2_ft"/></div>
<div  align="left"> <img src="./image/hidden_size_ft.png" width = "700" height = "392" alt="hidden_size_ft"/></div>


## We show BERT inference performance here.

<div  align="left"> <img src="./image/bert_ft.png" width = "700" height = "386" alt="bert_ft"/></div>
