from eet import pipeline
import torch
from PIL import Image

image_classifier = pipeline("zero-shot-image-classification",data_type = torch.float16)

image = Image.open("images/cat.jpg")
output = image_classifier(image, candidate_labels=["cat", "plane"])

print("outputs:",output)

