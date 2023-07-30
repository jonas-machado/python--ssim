import io
import os
import requests
import pandas as pd
from google.cloud import vision

os.environ['GOOGLE_APLICATION_CREDENTIALS'] = '.venv/sun-commerce-vision.json'
client = vision.ImageAnnotatorClient()


image_url = 'https://i.zst.com.br/thumbs/12/14/35/1365500211.jpg'
image = vision.Image()
image.source.image_uri = image_url

response = client.label_detection(image=image)
print(response)


