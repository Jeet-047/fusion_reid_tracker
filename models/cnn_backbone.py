import torch
import torchvision.models as models
import torch.nn as nn

class Identity(nn.Module):
    def forward(self, input):
        return input

def build_resnet50():
    model = models.resnet50(weights='IMAGENET1K_V1')
    model.fc = Identity()
    model.avgpool = Identity()
    # Override forward to return features before avgpool and flatten
    def forward_features(x):
        x = model.conv1(x)
        x = model.bn1(x)
        x = model.relu(x)
        x = model.maxpool(x)
        x = model.layer1(x)
        x = model.layer2(x)
        x = model.layer3(x)
        x = model.layer4(x)
        return x  # [B, 2048, 7, 7]
    model.forward = forward_features
    return model