#
# Created by djz on 2022/04/01.
#
from typing import List, Union
from transformers.utils import logging
from .base import  Pipeline
from PIL import Image
from transformers.image_utils import load_image
import torch

logger = logging.get_logger(__name__)


class ImageClassificationPipeline(Pipeline):
    def __init__(self, *args, **kwargs):
        super().__init__(*args, **kwargs)

    def _sanitize_parameters(self, top_k=None):
        postprocess_params = {}
        if top_k is not None:
            postprocess_params["top_k"] = top_k
        return {}, {}, postprocess_params

    def __call__(self, images: Union[str, List[str], "Image.Image", List["Image.Image"]], **kwargs):
        return super().__call__(images, **kwargs)

    def preprocess(self, image):
        image = load_image(image)
        model_inputs = self.feature_extractor(images=image, return_tensors='pt')
        return model_inputs

    def _forward(self, model_inputs):
        model_outputs = self.model(**model_inputs)
        return model_outputs

    def postprocess(self, model_outputs, top_k=5):
        if top_k > self.model.config.num_labels:
            top_k = self.model.config.num_labels
        if model_outputs.logits.dtype == torch.float16:
            model_outputs.logits = model_outputs.logits.to(dtype = torch.float32)
        probs = model_outputs.logits.softmax(-1)[0]
        scores, ids = probs.topk(top_k)

        scores = scores.tolist()
        ids = ids.tolist()
        return [{"score": score, "label": self.model.config.id2label[_id]} for score, _id in zip(scores, ids)]
