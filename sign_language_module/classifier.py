from models import get_model 
from utils import num2cat, load_image, get_gradcam_visualization 
import torch
import cv2
import numpy as np



model = get_model()


def predict(model, img):
    result = model(img)
    torch.argmax(result)
    result = num2cat[torch.argmax(result).item()]
    return result


if __name__ == '__main__':
    img = load_image("img/B.png")
    a = torch.squeeze(img, axis=0).numpy()
    
    result = predict(model, img)
    visualization = get_gradcam_visualization(model, img, result)

    import matplotlib.pyplot as plt
    plt.figure(figsize=(10, 10))
    plt.title(result.upper(), fontsize=56)
    plt.imshow(visualization)
    plt.show()

    print('result:', result)
    import time
    time.sleep(20000)
