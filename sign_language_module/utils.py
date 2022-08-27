from torchvision.models import ConvNeXt_Tiny_Weights
from torchvision import transforms
from PIL import Image
import torch
from pytorch_grad_cam import GradCAM 
from pytorch_grad_cam.utils.model_targets import ClassifierOutputTarget
from pytorch_grad_cam.utils.image import show_cam_on_image
import matplotlib.pyplot as plt
import numpy as np

inv_normalize = transforms.Normalize(
    mean=[-0.485/0.229, -0.456/0.224, -0.406/0.255],
    std=[1/0.229, 1/0.224, 1/0.255]
)

num2cat = {0: 'A',
           1: 'B', 
           2: 'C', 
           3: 'D', 
           4: 'E', 
           5: 'F', 
           6: 'G', 
           7: 'H', 
           8: 'I', 
           9: 'J', 
           10: 'K', 
           11: 'L', 
           12: 'M', 
           13: 'N', 
           14: 'O', 
           15: 'P', 
           16: 'Q', 
           17: 'R', 
           18: 'S', 
           19: 'T', 
           20: 'U', 
           21: 'V', 
           22: 'W', 
           23: 'X', 
           24: 'Y', 
           25: 'Z', 
           26: 'del', 
           27: 'nothing', 
           28: 'space'}
cat2num = {cat: i for i, cat in num2cat.items()}

transform = ConvNeXt_Tiny_Weights.IMAGENET1K_V1.transforms()


def load_image(imagepath):
  img = transform.forward(Image.open(imagepath).convert('RGB'))
  img = torch.unsqueeze(img, 0)
  return img

def visualize_gradcam(model, input_tensor, category):
    target_layers = [model.model.features[-1]]
    cam = GradCAM(model=model, target_layers=target_layers)

    targets = [ClassifierOutputTarget(cat2num[category])]

    # You can also pass aug_smooth=True and eigen_smooth=True, to apply smoothing.
    grayscale_cam = cam(input_tensor=input_tensor, targets=targets)

    # In this example grayscale_cam has only one image in the batch:
    grayscale_cam = grayscale_cam[0, :]

    img_rgb = torch.permute(inv_normalize.forward(input_tensor)[0], [1, 2, 0]).numpy()
    img_rgb = np.clip(img_rgb, 0, 1)
    visualization = show_cam_on_image(img_rgb, grayscale_cam, use_rgb=True)

    plt.figure(figsize=(10, 10))
    plt.title(category.upper(), fontsize=56)
    plt.imshow(visualization)
    plt.show()
