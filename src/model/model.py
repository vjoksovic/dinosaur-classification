import torch
import torch.nn as nn
import torch.nn.functional as F

class DinoNet(nn.Module):
    def __init__(self, num_classes: int, dropout: float = 0.5):
        super(DinoNet, self).__init__()

        # Feature extractor
        self.conv_layers = nn.Sequential(
            nn.Conv2d(3, 32, kernel_size=3, padding=1),  # 224x224 -> 224x224
            nn.BatchNorm2d(32),
            nn.ReLU(),
            nn.MaxPool2d(2, 2),  # 224 -> 112

            nn.Conv2d(32, 64, kernel_size=3, padding=1), # 112 -> 112
            nn.BatchNorm2d(64),
            nn.ReLU(),
            nn.MaxPool2d(2, 2),  # 112 -> 56

            nn.Conv2d(64, 128, kernel_size=3, padding=1), # 56 -> 56
            nn.BatchNorm2d(128),
            nn.ReLU(),
            nn.MaxPool2d(2, 2),  # 56 -> 28

            nn.Conv2d(128, 256, kernel_size=3, padding=1), # 28 -> 28
            nn.BatchNorm2d(256),
            nn.ReLU(),
            nn.MaxPool2d(2, 2),  # 28 -> 14
        )

        # Classifier
        self.fc_layers = nn.Sequential(
            nn.Linear(256 * 14 * 14, 512),
            nn.ReLU(),
            nn.Dropout(dropout),  # regularizacija
            nn.Linear(512, num_classes)
        )

    def forward(self, x):
        x = self.conv_layers(x)
        x = x.view(x.size(0), -1)  # flatten
        x = self.fc_layers(x)
        return x
