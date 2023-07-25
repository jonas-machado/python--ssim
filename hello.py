from skimage.metrics import structural_similarity
import cv2
import numpy as np
from flask import Flask, request, jsonify

app = Flask(__name__)

@app.route('/process_image', methods=['POST'])
def process_image():
    data = request.get_json()
    print(data)
    if 'image_url' not in data:
        return jsonify({'error': 'No URL provided in the request'}), 400
    
    image_url = data['image_url']
    print(image_url)
    before = cv2.imread(image_url)
    after = cv2.imread('assets/rg12005.webp')

    desired_width = 800 
    desired_height = 600

    #resized_image_before = cv2.resize(before, (desired_width, desired_height))
    resized_image_after = cv2.resize(after, (desired_width, desired_height))

    # Convert images to grayscale
    #before_gray = cv2.cvtColor(image_url, cv2.COLOR_BGR2GRAY)
    after_gray = cv2.cvtColor(resized_image_after, cv2.COLOR_BGR2GRAY)

    # Compute SSIM between two images
    (score, diff) = structural_similarity(image_url, after_gray, full=True)
    print("Image similarity", score)

# The diff image contains the actual image differences between the two images
# and is represented as a floating point data type in the range [0,1] 
# so we must convert the array to 8-bit unsigned integers in the range
# [0,255] before we can use it with OpenCV
    diff = (diff * 255).astype("uint8")

    # Threshold the difference image, followed by finding contours to
    # obtain the regions of the two input images that differ


    cv2.imshow('before', before)
    cv2.imshow('after', after)
    cv2.waitKey(0)

if __name__ == '__main__':
    app.run(debug=True)