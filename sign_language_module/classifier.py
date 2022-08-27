from models import get_model 
from utils import num2cat, load_image, visualize_gradcam
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
    # print(a)
    print(a.shape)
    # a = np.transpose(a, [1, 2, 0])
    # print(a)
    # cv2.imshow('img', a)
    result = predict(model, img)
    visualize_gradcam(model, img, result)

    print('result:', result)
    import time
    time.sleep(20000)



    
