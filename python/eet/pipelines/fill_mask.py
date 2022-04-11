#
# Created by djz on 2022/04/01.
#
import torch
import numpy as np
from typing import Dict
from transformers.utils import logging
from .base import GenericTensor,Pipeline,PipelineException
logger = logging.get_logger(__name__)

class FillMaskPipeline(Pipeline):

    def get_masked_index(self, input_ids: GenericTensor) -> np.ndarray:
        masked_index = torch.nonzero(input_ids == self.tokenizer.mask_token_id, as_tuple=False)
        return masked_index

    def _ensure_exactly_one_mask_token(self, input_ids: GenericTensor) -> np.ndarray:
        masked_index = self.get_masked_index(input_ids)
        numel = np.prod(masked_index.shape)
        if numel < 1:
            # raise ValueError("No mask_token ({self.tokenizer.mask_token}) found on the input")

            raise PipelineException(
                "fill-mask",
                self.model,
                f"No mask_token ({self.tokenizer.mask_token}) found on the input",
            )


    def ensure_exactly_one_mask_token(self, model_inputs: GenericTensor):
        if isinstance(model_inputs, list):
            for model_input in model_inputs:
                self._ensure_exactly_one_mask_token(model_input["input_ids"][0])
        else:
            for input_ids in model_inputs["input_ids"]:
                self._ensure_exactly_one_mask_token(input_ids)

    def preprocess(self, inputs, return_tensors=None, **preprocess_parameters) -> Dict[str, GenericTensor]:
        return_tensors = "pt"
        model_inputs = self.tokenizer(inputs,return_tensors = return_tensors)
        self.ensure_exactly_one_mask_token(model_inputs)
        return model_inputs

    def _forward(self, model_inputs):
        model_outputs = self.model(**model_inputs)
        model_outputs["input_ids"] = model_inputs["input_ids"]
        return model_outputs

    def postprocess(self, model_outputs, top_k=5, target_ids=None):
        # Cap top_k if there are targets
        if target_ids is not None and target_ids.shape[0] < top_k:
            top_k = target_ids.shape[0]
        input_ids = model_outputs["input_ids"][0]
        outputs = model_outputs["logits"]


        masked_index = torch.nonzero(input_ids == self.tokenizer.mask_token_id, as_tuple=False).squeeze(-1)
        # Fill mask pipeline supports only one ${mask_token} per sample

        logits = outputs[0, masked_index, :]
        if logits.dtype == torch.float16:
            logits = logits.to(dtype = torch.float32)
        probs = logits.softmax(dim=-1)
        if target_ids is not None:
            probs = probs[..., target_ids]

        values, predictions = probs.topk(top_k)

        result = []
        single_mask = values.shape[0] == 1
        for i, (_values, _predictions) in enumerate(zip(values.tolist(), predictions.tolist())):
            row = []
            for v, p in zip(_values, _predictions):
                # Copy is important since we're going to modify this array in place
                tokens = input_ids.numpy().copy()
                if target_ids is not None:
                    p = target_ids[p].tolist()

                tokens[masked_index[i]] = p
                # Filter padding out:
                tokens = tokens[np.where(tokens != self.tokenizer.pad_token_id)]
                # Originally we skip special tokens to give readable output.
                # For multi masks though, the other [MASK] would be removed otherwise
                # making the output look odd, so we add them back
                sequence = self.tokenizer.decode(tokens, skip_special_tokens=single_mask)
                proposition = {"score": v, "token": p, "token_str": self.tokenizer.decode(p), "sequence": sequence}
                row.append(proposition)
            result.append(row)
        if single_mask:
            return result[0]
        return result

    def get_target_ids(self, targets, top_k=None):
        if isinstance(targets, str):
            targets = [targets]
        try:
            vocab = self.tokenizer.get_vocab()
        except Exception:
            vocab = {}
        target_ids = []
        for target in targets:
            id_ = vocab.get(target, None)
            if id_ is None:
                input_ids = self.tokenizer(
                    target,
                    add_special_tokens=False,
                    return_attention_mask=False,
                    return_token_type_ids=False,
                    max_length=1,
                    truncation=True,
                )["input_ids"]
                if len(input_ids) == 0:
                    logger.warning(
                        f"The specified target token `{target}` does not exist in the model vocabulary. "
                        f"We cannot replace it with anything meaningful, ignoring it"
                    )
                    continue
                id_ = input_ids[0]
                # XXX: If users encounter this pass
                # it becomes pretty slow, so let's make sure
                # The warning enables them to fix the input to
                # get faster performance.
                logger.warning(
                    f"The specified target token `{target}` does not exist in the model vocabulary. "
                    f"Replacing with `{self.tokenizer.convert_ids_to_tokens(id_)}`."
                )
            target_ids.append(id_)
        target_ids = list(set(target_ids))
        if len(target_ids) == 0:
            raise ValueError("At least one target must be provided when passed.")
        target_ids = np.array(target_ids)
        return target_ids

    def _sanitize_parameters(self, top_k=None, targets=None):
        postprocess_params = {}

        if targets is not None:
            target_ids = self.get_target_ids(targets, top_k)
            postprocess_params["target_ids"] = target_ids

        if top_k is not None:
            postprocess_params["top_k"] = top_k

        if self.tokenizer.mask_token_id is None:
            raise PipelineException(
                "fill-mask", self.model.base_model_prefix, "The tokenizer does not define a `mask_token`."
            )
        return {}, {}, postprocess_params

    def __call__(self, inputs, *args, **kwargs):
        outputs = super().__call__(inputs, **kwargs)
        if isinstance(inputs, list) and len(inputs) == 1:
            return outputs[0]
        return outputs
