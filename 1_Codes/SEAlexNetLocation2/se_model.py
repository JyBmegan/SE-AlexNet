import torch
import torch.nn as nn
from torchvision import models

class SEBlock(nn.Module):
    """
    Squeeze-and-Excitation (SE) block for fully connected feature vectors.
    Given an input vector of size 'channel', it computes channel-wise scaling factors.
    """
    def __init__(self, channel, reduction=2):
        super(SEBlock, self).__init__()
        self.fc1 = nn.Linear(channel, channel // reduction, bias=True)
        self.relu = nn.ReLU(inplace=True)
        self.fc2 = nn.Linear(channel // reduction, channel, bias=True)
        self.sigmoid = nn.Sigmoid()
    
    def forward(self, x):
        # x shape: (batch_size, channel)
        y = self.fc1(x)
        y = self.relu(y)
        y = self.fc2(y)
        y = self.sigmoid(y)
        return x * y  # Element-wise multiplication

class AlexNetWithSE(nn.Module):
    """
    Modified AlexNet that inserts an SE block between fc6 and fc7.
    The classifier is altered such that after the first Linear (fc6) and ReLU,
    an SE block recalibrates the 4096 features before passing them to fc7.
    """
    def __init__(self, num_classes=8):
        super(AlexNetWithSE, self).__init__()
        # Use AlexNet's feature extractor from torchvision
        alexnet = models.alexnet(pretrained=False)
        self.features = alexnet.features
        self.avgpool = nn.AdaptiveAvgPool2d((6, 6))
        
        self.classifier = nn.Sequential(
            nn.Dropout(),
            nn.Linear(256 * 6 * 6, 4096),  # fc6
            nn.ReLU(inplace=True),
            # Insert SE block after fc6
            SEBlock(4096, reduction=16),
            nn.Dropout(),
            nn.Linear(4096, 4096),         # fc7
            nn.ReLU(inplace=True),
            nn.Linear(4096, num_classes),  # fc8
        )

    def forward(self, x):
        x = self.features(x)
        x = self.avgpool(x)
        x = torch.flatten(x, 1)
        x = self.classifier(x)
        return x