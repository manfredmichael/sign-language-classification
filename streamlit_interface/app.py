import base64
import json
import os
import re
import time
import uuid
from io import BytesIO, StringIO
from pathlib import Path

import numpy as np
import streamlit as st
from PIL import Image
# from streamlit_drawable_canvas import st_canvas
from utils import inference

def from_picture():
    st.image(Image.open('img/ASL_Alphabet.jpg'))
    
    st.markdown("### Upload a picture of ASL alphabet symbol")    
    webcam_file = st.file_uploader('Upload a picture', type=['jpg', 'jpeg', 'png'], accept_multiple_files=False)
    if webcam_file: 
        webcam = Image.open(webcam_file).convert('RGB')
        st.image(webcam)

    if webcam_file:
        st.markdown("---")
        st.markdown(f"# GradCam Visualization")

        result, confidence_score, visualization = inference(webcam)

        st.markdown(f"##### Prediction: {result}")
        st.markdown(f"Confidence: {confidence_score}")
        st.image(visualization, width=640)

def from_camera():
    st.image(Image.open('img/ASL_Alphabet.jpg'))
    
    st.markdown("### Take a picture of ASL alphabet symbol")    
    webcam_file = st.camera_input("Take a picture")
    if webcam_file: 
        webcam = Image.open(webcam_file).convert('RGB')
        st.image(webcam)

    if webcam_file:
        st.markdown("---")
        st.markdown(f"# GradCam Visualization")

        result, confidence_score, visualization = inference(webcam)

        st.markdown(f"##### Prediction: {result}")
        st.markdown(f"Confidence: {confidence_score}")
        st.image(visualization, width=640)

def main():
    page_names_to_funcs = {
        "From Uploaded Picture": from_picture,
        "From Camera": from_camera,
    }

    selected_page = st.sidebar.selectbox("Select a page", page_names_to_funcs.keys())
    page_names_to_funcs[selected_page]()

if __name__ == "__main__":
    st.set_page_config(
        page_title="Virtual Background with Image Processing", page_icon=":pencil2:"
    )
    st.title("Virtual Background with Image Processing")
    # st.sidebar.subheader("Configuration")
    main()
