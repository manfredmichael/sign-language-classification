from torchvision.models import ConvNeXt_Tiny_Weights

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
cat2num = {cat: i for i, cat in enumerate(class_names)}

transform = ConvNeXt_Tiny_Weights.IMAGENET1K_V1.transforms()

def load_image(imagepath):
  img = transform.forward(Image.open(imagepath).convert('RGB'))
  img = torch.unsqueeze(img, 0)
  return img

def predict(model, imagepath):
  img = load_image(imagepath)
  result = model(img)
  torch.argmax(result)
  result = num2cat[torch.argmax(result).item()]
  return result
