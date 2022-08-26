from .model import ASLModel
from utils import num2cat



model = ASLModel()

def predict(image):
    num = model([image])
    letter = num2cat[num]
    return letter

def __init__ == '__main__':
    
