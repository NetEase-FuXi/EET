为了适配fairseq训练出来的模型能够完美兼容pipelines，需要生成config.json和vocab.txt文件。

使用方法:

直接执行save_vocab_from_dict.py和save_config_from_fairseqmodel.py并修改保存路径即可，注意：保存路径要与预训练模型路径一致，对于生成vocab.txt，是将dict.txt转换过来的，需要在开头加上特殊字符，特殊字符与训练代码有关，此处提供的Special_Symbols.txt是网易伏羲自研的大模型特殊字符。

