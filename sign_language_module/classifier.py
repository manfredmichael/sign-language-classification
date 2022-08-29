from utils import num2cat, load_image, get_gradcam_visualization 
from models import get_model 
import torch
import numpy as np

model = get_model()

def predict(img, model=model, visualization=False):
    predictions = model(img)
    num_class = torch.argmax(predictions).item()
    probs = torch.nn.functional.softmax(predictions, dim=1)
    confidence_score = "{:.3f}".format(probs[0][num_class].item())
    result = num2cat[num_class]
    if visualization:
        visualization = get_gradcam_visualization(model, img, result)
        return result, confidence_score, visualization
    return result, confidence_score


if __name__ == '__main__':
    img = load_image("img/B.png")
    a = torch.squeeze(img, axis=0).numpy()
    
    result, confidence_score, visualization = predict(img, model=model, visualization=True)

    import matplotlib.pyplot as plt
    plt.figure(figsize=(10, 10))
    plt.title(f"{result.upper()} ({confidence_score})", fontsize=56)
    plt.imshow(visualization)
    plt.show()

    print('result:', result)
    import time
    time.sleep(20000)
