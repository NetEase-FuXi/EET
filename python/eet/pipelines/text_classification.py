#
# Created by djz on 2022/04/01.
#
import numpy as np
from typing import Dict
from transformers.file_utils import ExplicitEnum, add_end_docstrings, is_tf_available, is_torch_available
from .base import  GenericTensor, Pipeline

def sigmoid(_outputs):
    return 1.0 / (1.0 + np.exp(-_outputs))

def softmax(_outputs):
    maxes = np.max(_outputs, axis=-1, keepdims=True)
    shifted_exp = np.exp(_outputs - maxes)
    return shifted_exp / shifted_exp.sum(axis=-1, keepdims=True)


class ClassificationFunction(ExplicitEnum):
    SIGMOID = "sigmoid"
    SOFTMAX = "softmax"
    NONE = "none"

class TextClassificationPipeline(Pipeline):

    return_all_scores = False
    function_to_apply = ClassificationFunction.NONE

    def __init__(self, **kwargs):
        super().__init__(**kwargs)

    def _sanitize_parameters(self, return_all_scores=None, function_to_apply=None, **tokenizer_kwargs):
        preprocess_params = tokenizer_kwargs

        postprocess_params = {}
        if hasattr(self.model.config, "return_all_scores") and return_all_scores is None:
            return_all_scores = self.model.config.return_all_scores

        if return_all_scores is not None:
            postprocess_params["return_all_scores"] = return_all_scores

        if isinstance(function_to_apply, str):
            function_to_apply = ClassificationFunction[function_to_apply.upper()]

        if function_to_apply is not None:
            postprocess_params["function_to_apply"] = function_to_apply
        return preprocess_params, {}, postprocess_params

    def __call__(self, *args, **kwargs):
        result = super().__call__(*args, **kwargs)
        if isinstance(args[0], str):
            # This pipeline is odd, and return a list when single item is run
            return [result]
        else:
            return result

    def preprocess(self, inputs, **tokenizer_kwargs) -> Dict[str, GenericTensor]:
        return_tensors = 'pt'
        return self.tokenizer(inputs, return_tensors=return_tensors, **tokenizer_kwargs)

    def _forward(self, model_inputs):
        return self.model(**model_inputs)

    def postprocess(self, model_outputs, function_to_apply=None, return_all_scores=False):
        # Default value before `set_parameters`
        if function_to_apply is None:
            if self.model.config.problem_type == "multi_label_classification" or self.model.config.num_labels == 1:
                function_to_apply = ClassificationFunction.SIGMOID
            elif self.model.config.problem_type == "single_label_classification" or self.model.config.num_labels > 1:
                function_to_apply = ClassificationFunction.SOFTMAX
            elif hasattr(self.model.config, "function_to_apply") and function_to_apply is None:
                function_to_apply = self.model.config.function_to_apply
            else:
                function_to_apply = ClassificationFunction.NONE

        outputs = model_outputs["logits"][0]
        outputs = outputs.numpy()

        if function_to_apply == ClassificationFunction.SIGMOID:
            scores = sigmoid(outputs)
        elif function_to_apply == ClassificationFunction.SOFTMAX:
            scores = softmax(outputs)
        elif function_to_apply == ClassificationFunction.NONE:
            scores = outputs
        else:
            raise ValueError(f"Unrecognized `function_to_apply` argument: {function_to_apply}")

        if return_all_scores:
            return [{"label": self.model.config.id2label[i], "score": score.item()} for i, score in enumerate(scores)]
        else:
            return {"label": self.model.config.id2label[scores.argmax().item()], "score": scores.max().item()}
