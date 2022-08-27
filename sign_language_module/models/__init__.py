import torch
from models.ASLModel import ASLModel
from utils import num2cat

def get_model():
    weights = torch.load("checkpoints/ASL_model1.pt")
    model = ASLModel(num_classes=len(num2cat))
    model.load_state_dict(weights)
    return model
