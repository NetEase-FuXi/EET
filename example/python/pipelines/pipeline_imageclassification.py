import PIL
import torch
from PIL import Image
from eet import pipeline
from transformers import ViTFeatureExtractor
from eet import EETViTForImageClassification

us_model_api = False
us_pipeline = True
url = 'images/cat.jpg'

if us_model_api:
    image = PIL.Image.open(url)
    image = PIL.ImageOps.exif_transpose(image)
    image = image.convert("RGB")

    feature_extractor = ViTFeatureExtractor.from_pretrained('google/vit-base-patch16-224')
    model = EETViTForImageClassification.from_pretrained('google/vit-base-patch16-224',max_batch=1,data_type=torch.float32)
    inputs = feature_extractor(images=image, return_tensors="pt")
    inputs={key:inputs[key].cuda() for key in inputs}
    outputs = model(**inputs)
    logits = outputs.logits
    # model predicts one of the 1000 ImageNet classes
    predicted_class_idx = logits.argmax(-1).item()
    print("Predicted class:", model.config.id2label[predicted_class_idx])

if us_pipeline:
    image_classifier = pipeline("image-classification", model="google/vit-base-patch16-224",data_type = torch.float16)
    outputs = image_classifier([url])
    print("outputs:",outputs)
