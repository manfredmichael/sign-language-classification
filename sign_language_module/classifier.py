from models import get_model 
from utils import num2cat, load_image, get_gradcam_visualization 
import torch
import cv2
import numpy as np



model = get_model()


def predict(img, model=model, visualization=False):
    result = model(img)
    torch.argmax(result)
    result = num2cat[torch.argmax(result).item()]
    if visualization:
        visualization = get_gradcam_visualization(model, img, result)
        return result, visualization
    return result


if __name__ == '__main__':
    img = load_image("img/B.png")
    a = torch.squeeze(img, axis=0).numpy()
    
    result, visualization = predict(img, model=model, visualization=True)

    import matplotlib.pyplot as plt
    plt.figure(figsize=(10, 10))
    plt.title(result.upper(), fontsize=56)
    plt.imshow(visualization)
    plt.show()

    print('result:', result)
    import time
    time.sleep(20000)
