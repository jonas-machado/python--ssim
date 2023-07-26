from skimage.metrics import structural_similarity
import cv2
import numpy as np
import requests
from flask import Flask, request, jsonify
from flask_cors import CORS

app = Flask(__name__)
cors = CORS(app, resources={r"/processImage/*": {"origins": "*"}})

def orb_sim(img1, img2):
  # SIFT is no longer available in cv2 so using ORB
  orb = cv2.ORB_create()

  # detect keypoints and descriptors
  kp_a, desc_a = orb.detectAndCompute(img1, None)
  kp_b, desc_b = orb.detectAndCompute(img2, None)

  # define the bruteforce matcher object
  bf = cv2.BFMatcher(cv2.NORM_HAMMING, crossCheck=True)
    
  #perform matches. 
  matches = bf.match(desc_a, desc_b)
  #Look for similar regions with distance < 50. Goes from 0 to 100 so pick a number between.
  similar_regions = [i for i in matches if i.distance < 50]  
  if len(matches) == 0:
    return 0
  return len(similar_regions) / len(matches)

def structural_sim(img1, img2):

  sim, diff = structural_similarity(img1, img2, full=True)
  return sim

@app.route('/processImage', methods=['POST'])
def process_image():
    data = request.get_json()
    print(data)
    if 'imageUrl' not in data:
        return jsonify({'error': 'No URL provided in the request'}), 400
    
    image_url = data['imageUrl']
    image_url2 = data['imageUrl2']

    # Download image from the provided URL using requests
    image1 = requests.get(image_url)
    image2 = requests.get(image_url2)

    if image1.status_code != 200 or image2.status_code != 200:
        return jsonify({'error': 'Failed to download the image'}), 400
    
    # Convert image data to a NumPy array
    img_array = np.frombuffer(image1.content, dtype=np.uint8)
    before = cv2.imdecode(img_array, cv2.IMREAD_COLOR)

    # Assuming the 'assets/real3.jpg' file exists in the specified path
    img_array = np.frombuffer(image2.content, dtype=np.uint8)
    after = cv2.imdecode(img_array, cv2.IMREAD_COLOR)

    desired_width = 800
    desired_height = 600

    resized_image_before = cv2.resize(before, (desired_width, desired_height))
    resized_image_after = cv2.resize(after, (desired_width, desired_height))

    # Convert images to grayscale
    before_gray = cv2.cvtColor(resized_image_before, cv2.COLOR_BGR2GRAY)
    after_gray = cv2.cvtColor(resized_image_after, cv2.COLOR_BGR2GRAY)



    # Compute SSIM between two images
    orb = orb_sim(before_gray, after_gray) * 100
    ssim = structural_sim(before_gray, after_gray) * 100#1.0 means identical. Lower = not similar

    print("ORB similarity:", orb)
    print("ORB similarity:", ssim)

    return {
        "ORB": str(round(orb)),
        "SSIM": str(round(ssim))
        } # Convert score to a string before returning

if __name__ == '__main__':
    app.run(debug=True)
