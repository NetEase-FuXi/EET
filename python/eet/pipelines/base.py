#
# Created by djz on 2022/04/01.
#
import types
import torch
import warnings
import importlib
import collections
from packaging import version
from collections import UserDict
from abc import ABC, abstractmethod
from contextlib import contextmanager
from torch.utils.data import DataLoader, Dataset
from typing import Any, Dict, List, Optional, Tuple, Union
from transformers.feature_extraction_utils import PreTrainedFeatureExtractor
from transformers.file_utils import ModelOutput
from transformers.modelcard import ModelCard
from transformers.models.auto.configuration_auto import AutoConfig
from transformers.tokenization_utils import PreTrainedTokenizer
from transformers.utils import logging
from transformers.pipelines.pt_utils import (
    PipelineChunkIterator,
    PipelinePackIterator,
)
GenericTensor = Union[List["GenericTensor"], "torch.Tensor"]
logger = logging.get_logger(__name__)

def no_collate_fn(items):
    if len(items) != 1:
        raise ValueError("This collate_fn is meant to be used with batch_size=1")
    return items[0]


def _pad(items, key, padding_value, padding_side):
    batch_size = len(items)
    if isinstance(items[0][key], torch.Tensor):
        # Others include `attention_mask` etc...
        shape = items[0][key].shape
        dim = len(shape)
        if dim == 4:
            # This is probable image so padding shouldn't be necessary
            # B, C, H, W
            return torch.cat([item[key] for item in items], dim=0)
        max_length = max(item[key].shape[1] for item in items)
        dtype = items[0][key].dtype

        if dim == 2:
            tensor = torch.zeros((batch_size, max_length), dtype=dtype) + padding_value
        elif dim == 3:
            tensor = torch.zeros((batch_size, max_length, shape[-1]), dtype=dtype) + padding_value

        for i, item in enumerate(items):
            if dim == 2:
                if padding_side == "left":
                    tensor[i, -len(item[key][0]) :] = item[key][0].clone()
                else:
                    tensor[i, : len(item[key][0])] = item[key][0].clone()
            elif dim == 3:
                if padding_side == "left":
                    tensor[i, -len(item[key][0]) :, :] = item[key][0].clone()
                else:
                    tensor[i, : len(item[key][0]), :] = item[key][0].clone()
        return tensor
    else:
        return [item[key] for item in items]


def pad_collate_fn(tokenizer, feature_extractor):
    # Tokenizer
    t_padding_side = None
    # Feature extractor
    f_padding_side = None
    if tokenizer is None and feature_extractor is None:
        raise ValueError("Pipeline without tokenizer or feature_extractor cannot do batching")
    if tokenizer is not None:
        if tokenizer.pad_token_id is None:
            raise ValueError(
                "Pipeline with tokenizer without pad_token cannot do batching. You can try to set it with "
                "`pipe.tokenizer.pad_token_id = model.config.eos_token_id`."
            )
        else:
            t_padding_value = tokenizer.pad_token_id
            t_padding_side = tokenizer.padding_side
    if feature_extractor is not None:
        # Feature extractor can be images, where no padding is expected
        f_padding_value = getattr(feature_extractor, "padding_value", None)
        f_padding_side = getattr(feature_extractor, "padding_side", None)

    if t_padding_side is not None and f_padding_side is not None and t_padding_side != f_padding_side:
        raise ValueError(
            f"The feature extractor, and tokenizer don't agree on padding side {t_padding_side} != {f_padding_side}"
        )
    padding_side = "right"
    if t_padding_side is not None:
        padding_side = t_padding_side
    if f_padding_side is not None:
        padding_side = f_padding_side

    def inner(items):
        keys = set(items[0].keys())
        for item in items:
            if set(item.keys()) != keys:
                raise ValueError(
                    f"The elements of the batch contain different keys. Cannot batch them ({set(item.keys())} != {keys})"
                )
        # input_values, input_pixels, input_ids, ...
        padded = {}
        for key in keys:
            if key in {"input_ids"}:
                _padding_value = t_padding_value
            elif key in {"input_values", "pixel_values", "input_features"}:
                _padding_value = f_padding_value
            elif key in {"p_mask"}:
                _padding_value = 1
            elif key in {"attention_mask", "token_type_ids"}:
                _padding_value = 0
            else:
                # This is likely another random key maybe even user provided
                _padding_value = 0
            padded[key] = _pad(items, key, _padding_value, padding_side)
        return padded

    return inner


def infer_load_model(
    model,
    model_class = None,
    task: Optional[str] = None,
    max_batch = 1,
    full_seq_len = 512,
    data_type = torch.float32,
    **model_kwargs
):
    # 此处目前只支持单个模型
    if model_class is not None:
        config = AutoConfig.from_pretrained(model)
        if config.architectures[0] ==  "TransformerDecoder":
            # 这个时候不能用automodel,因为automodel是匹配的transformers的模型，此处需要单独处理
            transformers_module = importlib.import_module("eet")
            model_class = getattr(transformers_module, "EET" + config.architectures[0])
            model = model_class.from_pretrained(model_id_or_path = model,
                                                max_batch = max_batch,
                                                full_seq_len = full_seq_len, 
                                                data_type = data_type)
        else:
            model = model_class.from_pretrained(model_id_or_path = model,
                                            max_batch = max_batch,
                                            full_seq_len = full_seq_len, 
                                            data_type = data_type)
    elif isinstance(model, str):
        config = AutoConfig.from_pretrained(model)
        transformers_module = importlib.import_module("eet")
        model_class = getattr(transformers_module, "EET" + config.architectures[0])
        if config.model_type == "gpt2":
            model = model_class.from_pretrained(model_id_or_path = model,
                                                max_batch = max_batch,
                                                full_seq_len = full_seq_len, 
                                                data_type = data_type)
        else:
            model = model_class.from_pretrained(model_id_or_path = model,
                                    max_batch = max_batch,
                                    data_type = data_type)

    return model


def get_default_model(targeted_task: Dict, task_options: Optional[Any]) -> str:
    defaults = targeted_task["default"]
    if task_options:
        if task_options not in defaults:
            raise ValueError(f"The task does not provide any default models for options {task_options}")
        default_models = defaults[task_options]["model"]
    elif "model" in defaults:
        default_models = targeted_task["default"]["model"]
    else:
        # XXX This error message needs to be updated to be more generic if more tasks are going to become
        # parametrized
        raise ValueError('The task defaults can\'t be correctly selected. You probably meant "translation_XX_to_YY"')

    return default_models


class PipelineException(Exception):
    def __init__(self, task: str, model: str, reason: str):
        super().__init__(reason)

        self.task = task
        self.model = model


class ArgumentHandler(ABC):
    """
    Base interface for handling arguments for each [`~pipelines.Pipeline`].
    """

    @abstractmethod
    def __call__(self, *args, **kwargs):
        raise NotImplementedError()


class _ScikitCompat(ABC):
    """
    Interface layer for the Scikit and Keras compatibility.
    """

    @abstractmethod
    def transform(self, X):
        raise NotImplementedError()

    @abstractmethod
    def predict(self, X):
        raise NotImplementedError()

from transformers.pipelines.pt_utils import (
    PipelineDataset,
    PipelineIterator,
)


class Pipeline(_ScikitCompat):
    def __init__(
        self,
        model,
        model_class,
        tokenizer: Optional[PreTrainedTokenizer] = None,
        feature_extractor: Optional[PreTrainedFeatureExtractor] = None,
        modelcard: Optional[ModelCard] = None,
        task: str = "",
        args_parser: ArgumentHandler = None,
        device: int = 0,
        binary_output: bool = False,
        data_type = torch.float32,
        max_batch_size = 1,
        full_seq_len = 512,
        **kwargs,
    ):
        
        auto_model = infer_load_model(model = model,model_class = model_class,
                                    task = task,max_batch=max_batch_size,
                                    full_seq_len = full_seq_len,
                                    data_type = data_type)

        self.task = task
        self.model = auto_model
        self.tokenizer = tokenizer
        self.feature_extractor = feature_extractor
        self.modelcard = modelcard
        self.device = torch.device("cpu" if device < 0 else f"cuda:{device}")
        self.binary_output = binary_output

        # Update config with task specific parameters
        task_specific_params = self.model.config.task_specific_params
        if task_specific_params is not None and task in task_specific_params:
            self.model.config.update(task_specific_params.get(task))

        self.call_count = 0
        self._batch_size = kwargs.pop("batch_size", None)
        self._num_workers = kwargs.pop("num_workers", None)
        self._preprocess_params, self._forward_params, self._postprocess_params = self._sanitize_parameters(**kwargs)


    def transform(self, X):
        """
        Scikit / Keras interface to transformers' pipelines. This method will forward to __call__().
        """
        return self(X=X)

    def predict(self, X):
        """
        Scikit / Keras interface to transformers' pipelines. This method will forward to __call__().
        """
        return self(X=X)

    @contextmanager
    def device_placement(self):
        if self.device.type == "cuda":
            torch.cuda.set_device(self.device)

        yield

    def ensure_tensor_on_device(self, **inputs):
        return self._ensure_tensor_on_device(inputs, self.device)

    def _ensure_tensor_on_device(self, inputs, device):
        if isinstance(inputs, ModelOutput):
            return ModelOutput(
                {name: self._ensure_tensor_on_device(tensor, device) for name, tensor in inputs.items()}
            )
        elif isinstance(inputs, dict):
            return {name: self._ensure_tensor_on_device(tensor, device) for name, tensor in inputs.items()}
        elif isinstance(inputs, UserDict):
            return UserDict({name: self._ensure_tensor_on_device(tensor, device) for name, tensor in inputs.items()})
        elif isinstance(inputs, list):
            return [self._ensure_tensor_on_device(item, device) for item in inputs]
        elif isinstance(inputs, tuple):
            return tuple([self._ensure_tensor_on_device(item, device) for item in inputs])
        elif isinstance(inputs, torch.Tensor):
            return inputs.to(device)
        else:
            return inputs

    @abstractmethod
    def _sanitize_parameters(self, **pipeline_parameters):
        """
        _sanitize_parameters will be called with any excessive named arguments from either `__init__` or `__call__`
        methods. It should return 3 dictionnaries of the resolved parameters used by the various `preprocess`,
        `forward` and `postprocess` methods. Do not fill dictionnaries if the caller didn't specify a kwargs. This
        let's you keep defaults in function signatures, which is more "natural".

        It is not meant to be called directly, it will be automatically called and the final parameters resolved by
        `__init__` and `__call__`
        """
        raise NotImplementedError("_sanitize_parameters not implemented")

    @abstractmethod
    def preprocess(self, input_: Any, **preprocess_parameters: Dict) -> Dict[str, GenericTensor]:
        """
        Preprocess will take the `input_` of a specific pipeline and return a dictionnary of everything necessary for
        `_forward` to run properly. It should contain at least one tensor, but might have arbitrary other items.
        """
        raise NotImplementedError("preprocess not implemented")

    @abstractmethod
    def _forward(self, input_tensors: Dict[str, GenericTensor], **forward_parameters: Dict) -> ModelOutput:
        """
        _forward will receive the prepared dictionnary from `preprocess` and run it on the model. This method might
        involve the GPU or the CPU and should be agnostic to it. Isolating this function is the reason for `preprocess`
        and `postprocess` to exist, so that the hot path, this method generally can run as fast as possible.

        It is not meant to be called directly, `forward` is preferred. It is basically the same but contains additional
        code surrounding `_forward` making sure tensors and models are on the same device, disabling the training part
        of the code (leading to faster inference).
        """
        raise NotImplementedError("_forward not implemented")

    @abstractmethod
    def postprocess(self, model_outputs: ModelOutput, **postprocess_parameters: Dict) -> Any:
        """
        Postprocess will receive the raw outputs of the `_forward` method, generally tensors, and reformat them into
        something more friendly. Generally it will output a list or a dict or results (containing just strings and
        numbers).
        """
        raise NotImplementedError("postprocess not implemented")

    def get_inference_context(self):
        inference_context = (
            torch.inference_mode if version.parse(torch.__version__) >= version.parse("1.9.0") else torch.no_grad
        )
        return inference_context

    def forward(self, model_inputs, **forward_params):
        # with self.device_placement():
        #     inference_context = self.get_inference_context()
        #     with inference_context():
        model_inputs = self._ensure_tensor_on_device(model_inputs, device=self.device)
        model_outputs = self._forward(model_inputs, **forward_params)
        model_outputs = self._ensure_tensor_on_device(model_outputs, device=torch.device("cpu"))

        return model_outputs

    def get_iterator(
        self, inputs, num_workers: int, batch_size: int, preprocess_params, forward_params, postprocess_params
    ):
        if isinstance(inputs, collections.abc.Sized):
            dataset = PipelineDataset(inputs, self.preprocess, preprocess_params)
        else:
            if num_workers > 1:
                logger.warning(
                    "For iterable dataset using num_workers>1 is likely to result"
                    " in errors since everything is iterable, setting `num_workers=1`"
                    " to guarantee correctness."
                )
                num_workers = 1
            dataset = PipelineIterator(inputs, self.preprocess, preprocess_params)
        if "TOKENIZERS_PARALLELISM" not in os.environ:
            logger.info("Disabling tokenizer parallelism, we're using DataLoader multithreading already")
            os.environ["TOKENIZERS_PARALLELISM"] = "false"
        collate_fn = no_collate_fn if batch_size == 1 else pad_collate_fn(self.tokenizer, self.feature_extractor)
        dataloader = DataLoader(dataset, num_workers=num_workers, batch_size=batch_size, collate_fn=collate_fn)
        model_iterator = PipelineIterator(dataloader, self.forward, forward_params, loader_batch_size=batch_size)
        final_iterator = PipelineIterator(model_iterator, self.postprocess, postprocess_params)
        return final_iterator

    def __call__(self, inputs, *args, num_workers=None, batch_size=None, **kwargs):
        if args:
            logger.warning(f"Ignoring args : {args}")

        if num_workers is None:
            if self._num_workers is None:
                num_workers = 0
            else:
                num_workers = self._num_workers
        if batch_size is None:
            if self._batch_size is None:
                batch_size = 1
            else:
                batch_size = self._batch_size

        preprocess_params, forward_params, postprocess_params = self._sanitize_parameters(**kwargs)

        # Fuse __init__ params and __call__ params without modifying the __init__ ones.
        preprocess_params = {**self._preprocess_params, **preprocess_params}
        forward_params = {**self._forward_params, **forward_params}
        postprocess_params = {**self._postprocess_params, **postprocess_params}

        self.call_count += 1
        if self.call_count > 10 and self.device.type == "cuda":
            warnings.warn(
                "You seem to be using the pipelines sequentially on GPU. In order to maximize efficiency please use a dataset",
                UserWarning,
            )

        is_dataset = Dataset is not None and isinstance(inputs, Dataset)
        is_generator = isinstance(inputs, types.GeneratorType)
        is_list = isinstance(inputs, list)

        is_iterable = is_dataset or is_generator or is_list

        # TODO make the get_iterator work also for `tf` (and `flax`).

        if is_list:
            return self.run_multi(inputs, preprocess_params, forward_params, postprocess_params)
        elif is_iterable:
            return self.iterate(inputs, preprocess_params, forward_params, postprocess_params)
        else:
            return self.run_single(inputs, preprocess_params, forward_params, postprocess_params)

    def run_multi(self, inputs, preprocess_params, forward_params, postprocess_params):
        return [self.run_single(item, preprocess_params, forward_params, postprocess_params) for item in inputs]

    def run_single(self, inputs, preprocess_params, forward_params, postprocess_params):
        model_inputs = self.preprocess(inputs, **preprocess_params)
        model_outputs = self.forward(model_inputs, **forward_params)
        outputs = self.postprocess(model_outputs, **postprocess_params)
        return outputs

    def iterate(self, inputs, preprocess_params, forward_params, postprocess_params):
        # This function should become `get_iterator` again, this is a temporary
        # easy solution.
        for input_ in inputs:
            yield self.run_single(input_, preprocess_params, forward_params, postprocess_params)


class ChunkPipeline(Pipeline):
    def run_single(self, inputs, preprocess_params, forward_params, postprocess_params):
        all_outputs = []
        for model_inputs in self.preprocess(inputs, **preprocess_params):
            model_outputs = self.forward(model_inputs, **forward_params)
            all_outputs.append(model_outputs)
        outputs = self.postprocess(all_outputs, **postprocess_params)
        return outputs

    def get_iterator(
        self, inputs, num_workers: int, batch_size: int, preprocess_params, forward_params, postprocess_params
    ):
        if "TOKENIZERS_PARALLELISM" not in os.environ:
            logger.info("Disabling tokenizer parallelism, we're using DataLoader multithreading already")
            os.environ["TOKENIZERS_PARALLELISM"] = "false"
        if num_workers > 1:
            logger.warning(
                "For ChunkPipeline using num_workers>0 is likely to result in errors since everything is iterable, setting `num_workers=1` to guarantee correctness."
            )
            num_workers = 1
        dataset = PipelineChunkIterator(inputs, self.preprocess, preprocess_params)
        collate_fn = no_collate_fn if batch_size == 1 else pad_collate_fn(self.tokenizer, self.feature_extractor)
        dataloader = DataLoader(dataset, num_workers=num_workers, batch_size=batch_size, collate_fn=collate_fn)
        model_iterator = PipelinePackIterator(dataloader, self.forward, forward_params, loader_batch_size=batch_size)
        final_iterator = PipelineIterator(model_iterator, self.postprocess, postprocess_params)
        return final_iterator
