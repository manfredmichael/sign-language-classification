from io import BytesIO
from xmlrpc.server import resolve_dotted_attribute
from PIL import Image
from flask import Flask, request, send_file, jsonify
from base64 import decodebytes, encodebytes
import numpy as np
from classifier import predict 

app = Flask(__name__)

def get_response_image(image):
    return encode_image(Image.fromarray(np.uint8(image)))

def encode_image(pil_img):
    byte_arr = BytesIO()
    pil_img.save(byte_arr, format='PNG') # convert the PIL image to byte array
    encoded_img = encodebytes(byte_arr.getvalue()).decode('ascii') # encode as base64
    return encoded_img

def decode_image(image_bytes):
    image_bytes = decodebytes(image_bytes)
    image_bytes = BytesIO(image_bytes)
    image_bytes = Image.open(image_bytes)
    return np.array(image_bytes)

@app.route("/predict", methods=['POST'])
def make_prediction():
    if request.method == 'POST':
        webcam = request.files['webcam'].read()
        background = request.files['background'].read()
        virtual_background = request.files['virtual_background'].read()

        webcam = decode_image(webcam)
        background = decode_image(background)
        virtual_background = decode_image(virtual_background)

        result, visualization = predict(webcam, visualization=True)
        result = get_response_image(result)
        visualization = get_response_image(visualization)
        
        return jsonify({'result': result,
                        'visualization': visualization})

if __name__ == "__main__":
    app.run(host='0.0.0.0', port=5000)

