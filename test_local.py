import io
import os
import requests
import pandas as pd
from google.cloud import vision

os.environ['GOOGLE_APLICATION_CREDENTIALS'] = '.venv/sun-commerce-vision.json'
client = vision.ImageAnnotatorClient()

image_path = 'assets/rg1200.jpg'
with io.open(image_path, 'rb') as image_file:
    content = image_file.read()
image = vision.Image(content=content)

response = client.label_detection(image=image)
for label in response.label_annotations:
    print(label.description)

