import torch
import torch.nn as nn
import torch.nn.functional as F

class SEBlock(nn.Module):
    def __init__(self, in_channels, reduction=32):   # check step 2: the reduction of SE block
        super(SEBlock, self).__init__()
        # Squeeze operation: Global Average Pooling
        self.global_pool = nn.AdaptiveAvgPool2d(1)
        # Excitation operation: Fully connected layers
        self.fc1 = nn.Linear(in_channels, in_channels // reduction, bias=False)
        self.fc2 = nn.Linear(in_channels // reduction, in_channels, bias=False)
        self.sigmoid = nn.Sigmoid()

    def forward(self, x):
        # Squeeze: Global Average Pooling
        b, c, _, _ = x.size()
        y = self.global_pool(x).view(b, c)
        
        # Excitation: Two fully connected layers
        y = F.relu(self.fc1(y))
        y = self.fc2(y)
        
        # Scale (Excitation) and apply it to the input
        y = self.sigmoid(y).view(b, c, 1, 1)
        return x * y.expand_as(x)


class AlexNetWithSE(nn.Module):    # check step 1: the location of SE block    1 - first palce / 3 is already
    def __init__(self, num_classes=2):
        super(AlexNetWithSE, self).__init__()
        
        # AlexNet convolutional layers
        self.features = nn.Sequential(
            nn.Conv2d(3, 64, kernel_size=11, stride=4, padding=2),
            nn.ReLU(inplace=True),
            nn.MaxPool2d(kernel_size=3, stride=2),
            nn.Conv2d(64, 192, kernel_size=5, padding=2),
            nn.ReLU(inplace=True),
            nn.MaxPool2d(kernel_size=3, stride=2),
            nn.Conv2d(192, 384, kernel_size=3, padding=1),
            nn.ReLU(inplace=True),
            nn.Conv2d(384, 256, kernel_size=3, padding=1),
            nn.ReLU(inplace=True),
            nn.Conv2d(256, 256, kernel_size=3, padding=1),
            nn.ReLU(inplace=True),
            nn.MaxPool2d(kernel_size=3, stride=2)
        )
        
        # SE block inserted here between the last conv layer and the first linear layer
        self.se_block = SEBlock(256)  # Last conv layer output channels = 256
        
        # Fully connected layers
        self.classifier = nn.Sequential(
            nn.Dropout(),
            nn.Linear(9216, 4096),
            nn.ReLU(inplace=True),
            nn.Dropout(),
            nn.Linear(4096, 4096),
            nn.ReLU(inplace=True),
            nn.Linear(4096, 8)
        )

    def forward(self, x):
        # Pass through convolutional layers
        x = self.features(x)

        # print(f"Shape after extraction: {x.shape}")
        
        # Apply SE block before the classifier
        x = self.se_block(x)
        
        # print(f"Shape before flattening: {x.shape}")


        # Flatten the output for the fully connected layers
        x = x.view(x.size(0), -1)

        # print(f"Shape after flattening: {x.shape}")
        
        # Pass through classifier
        x = self.classifier(x)
        return x

