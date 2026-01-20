import torch
import torch.nn as nn
import torch.nn.functional as F

class SEBlock(nn.Module):
    def __init__(self, in_channels, reduction=16):   # check step 2: the reduction of SE block
        super(SEBlock, self).__init__()
        self.avg_pool = nn.AdaptiveAvgPool2d(1)
        self.fc1 = nn.Linear(in_channels, in_channels // reduction, bias=False)
        self.fc2 = nn.Linear(in_channels // reduction, in_channels, bias=False)
        self.sigmoid = nn.Sigmoid()

    def forward(self, x):
        # Squeeze: Global Average Pooling
        b, c, _, _ = x.size()
        y = self.avg_pool(x).view(b, c)
        
        # Excitation: Two fully connected layers
        y = F.relu(self.fc1(y))
        y = self.fc2(y)
        
        # Scale (Excitation) and apply it to the input
        y = self.sigmoid(y).view(b, c, 1, 1)
        return x * y.expand_as(x)


class AlexNetWithSE(nn.Module):  
    def __init__(self, num_classes = 8, se_pos = None):
        """
        We difine se_pos (int) as the insert location of SE module in Convolutional Layers (1-4):
        e.g.,
        se_pos (1) means insert SE Block after Conv1
        the fifth one should same as the SE-1 ?? like before the first Fc layer
        """
        super(AlexNetWithSE, self).__init__()
        
        layers = []

        # Condition 1: Conv1 (out 64)
        layers += [
            nn.Conv2d(3, 64, kernel_size=11, stride=4, padding=2),
            nn.ReLU(inplace=True),
            nn.MaxPool2d(kernel_size=3, stride=2)
        ]
        if se_pos == 1:
            layers.append(SEBlock(64))

        # Condition 2: Conv2 (out 192)
        layers += [
            nn.Conv2d(64, 192, kernel_size=5, padding=2),
            nn.ReLU(inplace=True),
            nn.MaxPool2d(kernel_size=3, stride=2)
        ]
        if se_pos == 2:
            layers.append(SEBlock(192))

        # Condition 3: Conv3 (out 384)
        layers += [
            nn.Conv2d(192, 384, kernel_size=3, padding=1),
            nn.ReLU(inplace=True),
        ]
        if se_pos == 3:
            layers.append(SEBlock(384))

        # Condition 4: Conv4 (out 256)
        layers += [
            nn.Conv2d(384, 256, kernel_size=3, padding=1),
            nn.ReLU(inplace=True),
        ]
        if se_pos == 4:
            layers.append(SEBlock(256))

        # Condition 5: Conv5 (out 256) = SE-1 (insert before fc6), So we wouldn't do it again
        layers += [
            nn.Conv2d(256, 256, kernel_size=3, padding=1),
            nn.ReLU(inplace=True),
            nn.MaxPool2d(kernel_size=3, stride=2)
        ]

        self.features = nn.Sequential(*layers)
        

        # Fully connected layers
        self.classifier = nn.Sequential(
            nn.Dropout(),
            nn.Linear(9216, 4096),
            nn.ReLU(inplace=True),
            nn.Dropout(),
            nn.Linear(4096, 4096),
            nn.ReLU(inplace=True),
            nn.Linear(4096, num_classes)
        )

    def forward(self, x):

        x = self.features(x)
        # print(f"Shape after extraction: {x.shape}")
        # x = self.se_block(x)
        # print(f"Shape before flattening: {x.shape}")
        x = x.view(x.size(0), -1)
        # print(f"Shape after flattening: {x.shape}")   
        # Pass through classifier
        x = self.classifier(x)
        return x

