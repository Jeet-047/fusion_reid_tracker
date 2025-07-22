import torch
from torchvision.models.vision_transformer import vit_b_16
def build_vit_b16():
    model = vit_b_16(pretrained=True)
    model.heads = torch.nn.Sequential()
    return model