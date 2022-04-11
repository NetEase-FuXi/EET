#
# Created by djz on 2022/04/01.
#
import os
import torch
from typing import Any, Dict, List, Optional, Tuple, Union
from transformers import pipeline as ts_pipeline
from .fill_mask import FillMaskPipeline
from .token_classification import TokenClassificationPipeline
from .question_answering import QuestionAnsweringPipeline
from .text_classification import TextClassificationPipeline
from .text_generation import TextGenerationPipeline
from .image_classification import ImageClassificationPipeline
from .zero_shot_image_classification import ZeroShotImageClassificationPipeline

from transformers.models.auto.feature_extraction_auto import FEATURE_EXTRACTOR_MAPPING, AutoFeatureExtractor
from transformers.models.auto.tokenization_auto import TOKENIZER_MAPPING, AutoTokenizer
from transformers import AutoConfig
from transformers import BertTokenizer
from transformers import logging
from .model_auto import (
            AutoModel,
            AutoModelForMaskedLM,
            AutoModelForCausalLM,
            AutoModelForSequenceClassification,
            AutoModelForTokenClassification,
            AutoModelForQuestionAnswering,
            AutoModelForImageClassification,
)

from .base import (
    Pipeline,
    get_default_model,
)
logger = logging.get_logger(__name__)

# Register all the supported tasks here
SUPPORTED_TASKS = {
    "text-classification": {
        "impl": TextClassificationPipeline,
        "pt": AutoModelForSequenceClassification,
        "default": {
            "model":"distilbert-base-uncased-finetuned-sst-2-english",
        },
        "type": "text",
    },
    "token-classification": {
        "impl": TokenClassificationPipeline,
        "pt": AutoModelForTokenClassification,
        "default": {
            "model": "dbmdz/bert-large-cased-finetuned-conll03-english",
        },
        "type": "text",
    },
    "question-answering": {
        "impl": QuestionAnsweringPipeline,
        "pt": AutoModelForQuestionAnswering,
        "default": {
            "model": "distilbert-base-cased-distilled-squad",
        },
        "type": "text",
    },
    "fill-mask": {
        "impl": FillMaskPipeline,
        "pt": AutoModelForMaskedLM,
        "default": {"model": "roberta-base"},
        "type": "text",
    },
    "text-generation": {
        "impl": TextGenerationPipeline,
        "pt": AutoModelForCausalLM,
        "default": {"model": "gpt2"},
        "type": "text",
    },
    "image-classification": {
        "impl": ImageClassificationPipeline,
        "pt": AutoModelForImageClassification,
        "default": {"model": "google/vit-base-patch16-224"},
        "type": "image",
    },
    "zero-shot-image-classification": {
        "impl": ZeroShotImageClassificationPipeline,
        "pt": AutoModel,
        "default": {"model": "openai/clip-vit-base-patch32"},
        "type": "multimodal",
    },
}

NO_FEATURE_EXTRACTOR_TASKS = set()
NO_TOKENIZER_TASKS = set()
for task, values in SUPPORTED_TASKS.items():
    if values["type"] == "text":
        NO_FEATURE_EXTRACTOR_TASKS.add(task)
    elif values["type"] in {"audio", "image"}:
        NO_TOKENIZER_TASKS.add(task)
    elif values["type"] != "multimodal":
        raise ValueError(f"SUPPORTED_TASK {task} contains invalid type {values['type']}")

def get_supported_tasks() -> List[str]:
    """
    Returns a list of supported task strings.
    """
    supported_tasks = list(SUPPORTED_TASKS.keys())
    supported_tasks.sort()
    return supported_tasks


def check_task(task: str) -> Tuple[Dict, Any]:
    if task in SUPPORTED_TASKS:
        targeted_task = SUPPORTED_TASKS[task]
        return targeted_task, None

    raise KeyError(f"Unknown task {task}, available tasks are {get_supported_tasks() + ['translation_XX_to_YY']}")

def pipeline(
    task: str = None,
    model = None,
    config = None,
    tokenizer = None,
    feature_extractor = None,
    framework = None,
    revision = None,
    use_fast = True,
    use_auth_token = None,
    model_kwargs = None,
    pipeline_class = None,
    data_type = torch.float32,
    us_ts = False,
    max_batch_size = 1,
    full_seq_len = 512,
    **kwargs
):
    if us_ts is False:
        if model_kwargs is None:
            model_kwargs = {}

        if task is None and model is None:
            raise RuntimeError(
                "Impossible to instantiate a pipeline without either a task or a model"
                "being specified."
                "Please provide a task class or a model"
            )

        if model is None and tokenizer is not None:
            raise RuntimeError(
                "Impossible to instantiate a pipeline with tokenizer specified but not the model "
                "as the provided tokenizer may not be compatible with the default model. "
                "Please provide a PreTrainedModel class or a path/identifier to a pretrained model when providing tokenizer."
            )
        if model is None and feature_extractor is not None:
            raise RuntimeError(
                "Impossible to instantiate a pipeline with feature_extractor specified but not the model "
                "as the provided feature_extractor may not be compatible with the default model. "
                "Please provide a PreTrainedModel class or a path/identifier to a pretrained model when providing feature_extractor."
            )
        targeted_task, task_options = check_task(task)
        if pipeline_class is None:
            pipeline_class = targeted_task["impl"]
            
        model_class = targeted_task["pt"]

        # Use default model/config/tokenizer for the task if no model is provided
        if model is None:
            # At that point framework might still be undetermined
            model = get_default_model(targeted_task, task_options)
            logger.warning(f"No model was supplied, defaulted to {model} (https://huggingface.co/{model})")

        model_config = AutoConfig.from_pretrained(model)
        # model_config = auto_model_base.config
        load_tokenizer = type(model_config) in TOKENIZER_MAPPING or model_config.tokenizer_class is not None
        load_feature_extractor = type(model_config) in FEATURE_EXTRACTOR_MAPPING or feature_extractor is not None
        if task in NO_TOKENIZER_TASKS:
            load_tokenizer = False
        if task in NO_FEATURE_EXTRACTOR_TASKS:
            load_feature_extractor = False

        if load_tokenizer:
            # Try to infer tokenizer from model or config name (if provided as str)
            if tokenizer is None:
                if isinstance(model, str):
                    tokenizer = model
                elif isinstance(config, str):
                    tokenizer = config
                else:
                    # Impossible to guess what is the right tokenizer here
                    raise Exception(
                        "Impossible to guess which tokenizer to use. "
                        "Please provide a PreTrainedTokenizer class or a path/identifier to a pretrained tokenizer."
                    )

            # Instantiate tokenizer if needed
            if isinstance(tokenizer, (str, tuple)):
                if isinstance(tokenizer, tuple):
                    # For tuple we have (tokenizer name, {kwargs})
                    use_fast = tokenizer[1].pop("use_fast", use_fast)
                    tokenizer_identifier = tokenizer[0]
                    tokenizer_kwargs = tokenizer[1]
                else:
                    tokenizer_identifier = tokenizer
                    tokenizer_kwargs = model_kwargs

                if model_config.architectures[0] ==  "TransformerDecoder":
                    # todo
                    # 用BertTokenizer是因为fairseq训练的模型是用的字符级别的编码
                    logger.info("This is a model trained with fairseq!")
                    tokenizer = BertTokenizer(vocab_file=tokenizer_identifier+'/vocab.txt',
                                bos_token='<s>', eos_token='</s>',
                                unk_token='<unk>', sep_token='<s>', pad_token='<pad>', cls_token='<cls>',
                                mask_token='<mask>')
                else:
                    tokenizer = AutoTokenizer.from_pretrained(
                        tokenizer_identifier, revision=revision, use_fast=use_fast, _from_pipeline=task, **tokenizer_kwargs
                    )



        if load_feature_extractor:
            # Try to infer feature extractor from model or config name (if provided as str)
            if feature_extractor is None:
                if isinstance(model, str):
                    feature_extractor = model
                elif isinstance(config, str):
                    feature_extractor = config
                else:
                    # Impossible to guess what is the right feature_extractor here
                    raise Exception(
                        "Impossible to guess which feature extractor to use. "
                        "Please provide a PreTrainedFeatureExtractor class or a path/identifier "
                        "to a pretrained feature extractor."
                    )

            # Instantiate feature_extractor if needed
            if isinstance(feature_extractor, (str, tuple)):
                feature_extractor = AutoFeatureExtractor.from_pretrained(
                    feature_extractor, revision=revision, _from_pipeline=task, **model_kwargs
                )

                if (
                    feature_extractor._processor_class
                    and feature_extractor._processor_class.endswith("WithLM")
                    and isinstance(model, str)
                ):
                    try:
                        import kenlm  # to trigger `ImportError` if not installed
                        from pyctcdecode import BeamSearchDecoderCTC

                        if os.path.isdir(model) or os.path.isfile(model):
                            decoder = BeamSearchDecoderCTC.load_from_dir(model)
                        else:
                            language_model_glob = os.path.join(
                                BeamSearchDecoderCTC._LANGUAGE_MODEL_SERIALIZED_DIRECTORY, "*"
                            )
                            alphabet_filename = BeamSearchDecoderCTC._ALPHABET_SERIALIZED_FILENAME
                            allow_regex = [language_model_glob, alphabet_filename]
                            decoder = BeamSearchDecoderCTC.load_from_hf_hub(model, allow_regex=allow_regex)

                        kwargs["decoder"] = decoder
                    except ImportError as e:
                        logger.warning(
                            f"Could not load the `decoder` for {model}. Defaulting to raw CTC. Try to install `pyctcdecode` and `kenlm`: (`pip install pyctcdecode`, `pip install https://github.com/kpu/kenlm/archive/master.zip`): Error: {e}"
                        )


        if tokenizer is not None:
            kwargs["tokenizer"] = tokenizer

        if feature_extractor is not None:
            kwargs["feature_extractor"] = feature_extractor
        # full_seq_len提示词最长的长度，只有GPT模型做生成才生效。
        return pipeline_class(model_class = model_class,
                            model=model, 
                            task=task,
                            data_type = data_type,
                            max_batch_size = max_batch_size,
                            full_seq_len = full_seq_len ,
                            **kwargs)
    else:
        return ts_pipeline(task = task,
            model = model,
            config = config,
            tokenizer = tokenizer,
            feature_extractor = feature_extractor,
            framework = framework,
            revision = revision,
            use_fast = use_fast,
            use_auth_token = use_auth_token,
            model_kwargs = model_kwargs,
            pipeline_class = pipeline_class,
            **kwargs
        )
