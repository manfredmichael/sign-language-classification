# sign-language-classification
Sign Language Classifier using ConvNeXt for my week 2 project during Nodeflux internship.

![](https://github.com/manfredmichael/sign-language-classification/blob/main/img/img1.png?raw=true)

## How to Run This App

### Via Cloning This Repo
First, clone this repo: `https://github.com/manfredmichael/sign-language-classification.git`. Please ensure you have [installed Docker](https://docs.docker.com/engine/install/ubuntu/). Next, you need to run both inference API & streamlit interface to use this demo.

##### 1. Run inference API
- Change the working directory: `cd sign_language_module`
- Build docker image: `docker build -t asl_app .`
- Run docker image: `docker run --rm -p 5000:5000 asl_app`

##### 2. Run streamlit interface
- Change the working directory: `cd streamlit_interface`
- Install dependenies: `pip install -r requirements.txt`
- Run streamlit app: `streamlit run app.py`


### Technologies & Tools
![](https://github.com/manfredmichael/sign-language-classification/blob/main/img/img2.png?raw=true)

### About the model

[ConvNeXt](https://arxiv.org/abs/2201.03545) Tiny is retrained on [American Sign Language Alphabet Dataset](https://www.kaggle.com/datasets/grassknoted/asl-alphabet) with 87,000 images (200x200) of 29 classes: A-Z & SPACE, DELETE, NOTHING.
