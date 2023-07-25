from skimage.metrics import structural_similarity
import cv2
import numpy as np
import requests
from flask import Flask, request, jsonify
from flask_cors import CORS

app = Flask(__name__)
CORS(app)

@app.route('/processImage', methods=['POST'])
def process_image():
    data = request.get_json()
    print(data)
    if 'image_url' not in data:
        return jsonify({'error': 'No URL provided in the request'}), 400
    
    image_url = data['imageUrl']
    print(image_url)
    # Download image from the provided URL using requests
    response = requests.get(image_url)
    if response.status_code != 200:
        return jsonify({'error': 'Failed to download the image'}), 400
    
    # Convert image data to a NumPy array
    img_array = np.frombuffer(response.content, dtype=np.uint8)
    before = cv2.imdecode(img_array, cv2.IMREAD_COLOR)

    # Assuming the 'assets/real3.jpg' file exists in the specified path
    after = cv2.imread('assets/jaq.jpg')

    desired_width = 800
    desired_height = 600

    resized_image_before = cv2.resize(before, (desired_width, desired_height))
    resized_image_after = cv2.resize(after, (desired_width, desired_height))

    # Convert images to grayscale
    before_gray = cv2.cvtColor(resized_image_before, cv2.COLOR_BGR2GRAY)
    after_gray = cv2.cvtColor(resized_image_after, cv2.COLOR_BGR2GRAY)

    # Compute SSIM between two images
    (score, diff) = structural_similarity(before_gray, after_gray, full=True)
    print("Image similarity:", score)

    return "Image similarity: " + str(score)  # Convert score to a string before returning

if __name__ == '__main__':
    app.run(debug=True)
