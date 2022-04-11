#
# Created by djz on 2022/04/01.
#
import importlib
from collections import OrderedDict
from transformers import AutoConfig

MODEL_MAPPING_NAMES = OrderedDict(
    [
        # Base model mapping
        ("roberta", "EETRobertaModel"),
        ("bert", "EETBertModel"),
        ("albert", "EETAlbertModel"),
        ("gpt2", "EETGPT2Model"),
        ("distilbert", "EETDistilBertModel"),
        ("vit", "EETViTModel"),
        ("clip", "EETCLIPModel"),
    ]
)

MODEL_FOR_MASKED_LM_MAPPING_NAMES = OrderedDict(
    [
        # Model for Masked LM mapping
        ("albert", "EETAlbertForMaskedLM"),
        ("roberta", "EETRobertaForMaskedLM"),
        ("bert", "EETBertForMaskedLM"),
        ("distilbert", "EETDistilBertForMaskedLM"),
    ]
)

MODEL_FOR_QUESTION_ANSWERING_MAPPING_NAMES = OrderedDict(
    [
        # Model for Question Answering mapping
        ("albert", "EETAlbertForQuestionAnswering"),
        ("roberta", "EETRobertaForQuestionAnswering"),
        ("bert", "EETBertForQuestionAnswering"),
        ("distilbert", "EETDistilBertForQuestionAnswering"),
    ]
)

MODEL_FOR_TOKEN_CLASSIFICATION_MAPPING_NAMES = OrderedDict(
    [
        # Model for Token Classification mapping
        ("roberta", "EETRobertaForTokenClassification"),
        ("bert", "EETBertForTokenClassification"),
        ("albert", "EETAlbertForTokenClassification"),
        ("gpt2", "EETGPT2ForTokenClassification"),
        ("distilbert", "EETDistilBertForTokenClassification"),
    ]
)

MODEL_FOR_MULTIPLE_CHOICE_MAPPING_NAMES = OrderedDict(
    [
        # Model for Multiple Choice mapping
        ("roberta", "EETRobertaForMultipleChoice"),
        ("bert", "EETBertForMultipleChoice"),
        ("albert", "EETAlbertForMultipleChoice"),
        ("distilbert", "EETDistilBertForMultipleChoice"),
    ]
)

MODEL_FOR_NEXT_SENTENCE_PREDICTION_MAPPING_NAMES = OrderedDict(
    [
        ("bert", "EETBertForNextSentencePrediction"),
    ]
)

MODEL_FOR_CAUSAL_LM_MAPPING_NAMES = OrderedDict(
    [
        # Model for Causal LM mapping
        ("roberta", "EETRobertaForCausalLM"),
        ("bert", "EETBertLMHeadModel"),
        ("gpt2", "EETGPT2LMHeadModel"),
        # ("gpt2", "EETTransformerDecoder"),
    ]
)


MODEL_FOR_SEQUENCE_CLASSIFICATION_MAPPING_NAMES = OrderedDict(
    [
        # Model for Sequence Classification mapping
        ("albert", "EETAlbertForSequenceClassification"),
        ("roberta", "EETRobertaForSequenceClassification"),
        ("bert", "EETBertForSequenceClassification"),
        ("gpt2", "EETGPT2ForSequenceClassification"),
        ("distilbert", "EETDistilBertForSequenceClassification"),
    ]
)
MODEL_WITH_LM_HEAD_MAPPING_NAMES = OrderedDict(
    [
        # Model with LM heads mapping
        ("albert", "EETAlbertForMaskedLM"),
        ("roberta", "EETRobertaForMaskedLM"),
        ("bert", "EETBertForMaskedLM"),
        ("gpt2", "EETGPT2LMHeadModel"),
        ("distilbert", "EETDistilBertForMaskedLM"),
    ]
)

MODEL_FOR_PRETRAINING_MAPPING_NAMES = OrderedDict(
    [
        # Model for pre-training mapping
        ("albert", "EETAlbertForPreTraining"),
        ("roberta", "EETRobertaForMaskedLM"),
        ("bert", "EETBertForPreTraining"),
        ("gpt2", "EETGPT2LMHeadModel"),
        ("distilbert", "EETDistilBertForMaskedLM"),
    ]
)

MODEL_FOR_MASKED_IMAGE_MODELING_MAPPING_NAMES = OrderedDict(
    [
        ("vit", "EETViTForMaskedImageModeling"),
    ]
)

MODEL_FOR_IMAGE_CLASSIFICATION_MAPPING_NAMES = OrderedDict(
    [
        ("vit", "EETViTForImageClassification"),
    ]
)

class AutoModelBase():
    @classmethod
    def from_pretrained(cls,model_id_or_path: str,max_batch, full_seq_len,data_type,model_OrderedDict):
        config = AutoConfig.from_pretrained(model_id_or_path)
        model_type = config.model_type
        if (model_type in model_OrderedDict) is False:
            raise RuntimeError(
                "EET does not support " + model_type
            )
        else:
            model_name = model_OrderedDict[model_type]
            transformers_module = importlib.import_module("eet")
            auto_model = getattr(transformers_module, model_name)
            if model_type == 'gpt2':
                return auto_model.from_pretrained(model_id_or_path, max_batch, full_seq_len,data_type)
            else:
                return auto_model.from_pretrained(model_id_or_path, max_batch,data_type)

class AutoModel(AutoModelBase):
    @classmethod
    def from_pretrained(cls,model_id_or_path: str,max_batch,data_type,full_seq_len = 512):
        return super().from_pretrained(model_id_or_path, max_batch,full_seq_len, data_type,MODEL_MAPPING_NAMES)

class AutoModelForMaskedLM(AutoModelBase):
    @classmethod
    def from_pretrained(cls,model_id_or_path: str,max_batch,data_type,full_seq_len = 512):
        return super().from_pretrained(model_id_or_path, max_batch,full_seq_len, data_type,MODEL_WITH_LM_HEAD_MAPPING_NAMES)


class AutoModelForPreTraining(AutoModelBase):
    @classmethod
    def from_pretrained(cls,model_id_or_path: str,max_batch, data_type,full_seq_len = 512):
        return super().from_pretrained(model_id_or_path, max_batch, full_seq_len, data_type,MODEL_FOR_PRETRAINING_MAPPING_NAMES)


class AutoModelForCausalLM(AutoModelBase):
    @classmethod
    def from_pretrained(cls,model_id_or_path: str,max_batch, data_type,full_seq_len = 512):
        return super().from_pretrained(model_id_or_path, max_batch, full_seq_len, data_type,MODEL_FOR_CAUSAL_LM_MAPPING_NAMES)



class AutoModelForNextSentencePrediction(AutoModelBase):
    @classmethod
    def from_pretrained(cls,model_id_or_path: str,max_batch, data_type,full_seq_len = 512):
        return super().from_pretrained(model_id_or_path, max_batch,full_seq_len, data_type,MODEL_FOR_NEXT_SENTENCE_PREDICTION_MAPPING_NAMES)


class AutoModelForSequenceClassification(AutoModelBase):
    @classmethod
    def from_pretrained(cls,model_id_or_path: str,max_batch,data_type,full_seq_len = 512):
        return super().from_pretrained(model_id_or_path, max_batch,full_seq_len, data_type,MODEL_FOR_SEQUENCE_CLASSIFICATION_MAPPING_NAMES)

class AutoModelForMultipleChoice(AutoModelBase):
    @classmethod
    def from_pretrained(cls,model_id_or_path: str,max_batch,data_type,full_seq_len = 512):
        return super().from_pretrained(model_id_or_path, max_batch,full_seq_len, data_type,MODEL_FOR_MULTIPLE_CHOICE_MAPPING_NAMES)

class AutoModelForTokenClassification(AutoModelBase):
    @classmethod
    def from_pretrained(cls,model_id_or_path: str,max_batch,data_type,full_seq_len = 512):
        return super().from_pretrained(model_id_or_path, max_batch,full_seq_len, data_type,MODEL_FOR_TOKEN_CLASSIFICATION_MAPPING_NAMES)

class AutoModelForQuestionAnswering(AutoModelBase):
    @classmethod
    def from_pretrained(cls,model_id_or_path: str,max_batch,data_type,full_seq_len = 512):
        return super().from_pretrained(model_id_or_path, max_batch,full_seq_len, data_type,MODEL_FOR_QUESTION_ANSWERING_MAPPING_NAMES)

class AutoModelForImageClassification(AutoModelBase):
    @classmethod
    def from_pretrained(cls,model_id_or_path: str,max_batch,data_type,full_seq_len = 512):
        return super().from_pretrained(model_id_or_path, max_batch,full_seq_len, data_type,MODEL_FOR_IMAGE_CLASSIFICATION_MAPPING_NAMES)

class AutoModelForMaskedImageModeling(AutoModelBase):
    @classmethod
    def from_pretrained(cls,model_id_or_path: str,max_batch,data_type,full_seq_len = 512):
        return super().from_pretrained(model_id_or_path, max_batch,full_seq_len, data_type,MODEL_FOR_MASKED_IMAGE_MODELING_MAPPING_NAMES)
