import requests
import json
from PIL import Image
from io import BytesIO, StringIO
from base64 import decodebytes, encodebytes
import numpy as np
import os
import dotenv

dotenv.load_dotenv()

inference_url = os.getenv('INFERENCE_URL') 

def encode_image(pil_img):
    byte_arr = BytesIO()
    pil_img.save(byte_arr, format='PNG') # convert the PIL image to byte array
    encoded_img = encodebytes(byte_arr.getvalue()).decode('ascii') # encode as base64
    return encoded_img

def decode_image(image_bytes):
    image_bytes = image_bytes.encode('ascii')
    image_bytes = decodebytes(image_bytes)
    image_bytes = BytesIO(image_bytes)
    image_bytes = Image.open(image_bytes)
    return np.array(image_bytes)

def inference(webcam):
    
    response = requests.post(
            f"{inference_url}/predict",
        files = {'webcam': encode_image(webcam)},
    )
    response = response.json()
    result = response['result']
    confidence_score = response['confidence_score']
    visualization = decode_image(response['visualization'])

    return result, confidence_score, visualization
