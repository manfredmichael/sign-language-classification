FROM python:3.10-slim-buster
WORKDIR /app

COPY requirements.txt app/
RUN pip install -r app/requirements.txt
RUN pip install opencv-python-headless

COPY checkpoints/ .
COPY checkpoints/ASL_model.pt checkpoints/ASL_model.pt 
COPY models/ .
COPY models/__init__.py models/__init__.py
COPY models/ASLModel.py models/ASLModel.py
COPY utils.py .
COPY classifier.py .
COPY app.py .

ENTRYPOINT [ "python" ]
CMD ["app.py" ]

EXPOSE 5000
