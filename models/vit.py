import torch
import torch.nn as nn
from transformers import ViTImageProcessor, ViTModel

class VIT(nn.Module):
    def __init__(self, num_classes=10, C=3, H=32, W=32, T=4):
        super().__init__()

        processor = ViTImageProcessor.from_pretrained('google/vit-base-patch16-224-in21k')
        model = ViTModel.from_pretrained('google/vit-base-patch16-224-in21k')

        


