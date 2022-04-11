import torch
from PIL import Image
from typing import List, Union
from .base import ChunkPipeline
from transformers.utils import logging
from transformers.image_utils import load_image
logger = logging.get_logger(__name__)


class ZeroShotImageClassificationPipeline(ChunkPipeline):
    def __init__(self, **kwargs):
        super().__init__(**kwargs)

    def __call__(self, images: Union[str, List[str], "Image", List["Image"]], **kwargs):
        return super().__call__(images, **kwargs)

    def _sanitize_parameters(self, **kwargs):
        preprocess_params = {}
        if "candidate_labels" in kwargs:
            preprocess_params["candidate_labels"] = kwargs["candidate_labels"]
        if "hypothesis_template" in kwargs:
            preprocess_params["hypothesis_template"] = kwargs["hypothesis_template"]

        return preprocess_params, {}, {}

    def preprocess(self, image, candidate_labels=None, hypothesis_template="This is a photo of {}."):
        n = len(candidate_labels)
        for i, candidate_label in enumerate(candidate_labels):
            image = load_image(image)
            images = self.feature_extractor(images=[image], return_tensors='pt')
            sequence = hypothesis_template.format(candidate_label)
            inputs = self.tokenizer(sequence, return_tensors='pt')
            inputs["pixel_values"] = images.pixel_values
            yield {"is_last": i == n - 1, "candidate_label": candidate_label, **inputs}

    def _forward(self, model_inputs):
        is_last = model_inputs.pop("is_last")
        candidate_label = model_inputs.pop("candidate_label")
        outputs = self.model(**model_inputs)

        # Clip does crossproduct scoring by default, so we're only
        # interested in the results where image and text and in the same
        # batch position.
        diag = torch.diagonal
        logits_per_image = diag(outputs[0])

        model_outputs = {
            "is_last": is_last,
            "candidate_label": candidate_label,
            "logits_per_image": logits_per_image,
        }
        return model_outputs

    def postprocess(self, model_outputs):
        candidate_labels = [outputs["candidate_label"] for outputs in model_outputs]
        logits = torch.cat([output["logits_per_image"] for output in model_outputs])
        if logits.dtype == torch.float16:
            logits = logits.to(dtype = torch.float32)

        probs = logits.softmax(dim=0)
        scores = probs.tolist()

        result = [
            {"score": score, "label": candidate_label}
            for score, candidate_label in sorted(zip(scores, candidate_labels), key=lambda x: -x[0])
        ]
        return result
