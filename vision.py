import io
import os
import pandas as pd
import json
from google.cloud import vision
from flask import Flask, request, jsonify
from flask_cors import CORS

app = Flask(__name__)
cors = CORS(app, resources={r"/processImage/*": {"origins": "*"}})

os.environ['GOOGLE_APLICATION_CREDENTIALS'] = '.venv/sun-commerce-vision.json'
client = vision.ImageAnnotatorClient()

@app.route('/processImageVision', methods=['POST'])
def process_image():
    data = request.get_json()
    if 'imageUrl' not in data:
        return jsonify({'error': 'No URL provided in the request'}), 400
    
    image_url = data['imageUrl']
    print(image_url)
    image = vision.Image()
    image.source.image_uri = image_url
    response = client.label_detection(image=image)
    return jsonify({
        "label": response.label_annotations[1]
    })

if __name__ == '__main__':
    app.run(debug=True)
