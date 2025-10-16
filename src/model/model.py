import torch
import torch.nn as nn
import torch.nn.functional as F
from torchvision import models

class DinoNet(nn.Module):
    def __init__(self, num_classes: int, dropout: float = 0.5, pretrained: bool = False):
        super(DinoNet, self).__init__()

        # ResNet-34 backbone
        weights = models.ResNet34_Weights.DEFAULT if pretrained else None
        self.backbone = models.resnet34(weights=weights)
        in_features = self.backbone.fc.in_features
        # Replace final classification head
        self.backbone.fc = nn.Sequential(
            nn.Dropout(p=dropout),
            nn.Linear(in_features, num_classes),
        )

    def forward(self, x):
        return self.backbone(x)
