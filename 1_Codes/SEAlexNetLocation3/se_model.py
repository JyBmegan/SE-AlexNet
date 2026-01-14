import torch.nn as nn
import torch


class AlexNetWithSE(nn.Module):
    def __init__(self, num_classes=1000, init_weights=False):
        super(AlexNetWithSE, self).__init__()
        self.features = nn.Sequential(
            nn.Conv2d(3, 64, kernel_size=11, stride=4, padding=2),  # input[3, 224, 224]  output[48, 55, 55]
            nn.ReLU(inplace=True),
            nn.MaxPool2d(kernel_size=3, stride=2),                  # output[48, 27, 27]
            nn.Conv2d(64, 192, kernel_size=5, padding=2),           # output[128, 27, 27]
            nn.ReLU(inplace=True),
            nn.MaxPool2d(kernel_size=3, stride=2),                  # output[128, 13, 13]
            nn.Conv2d(192, 384, kernel_size=3, padding=1),          # output[192, 13, 13]
            nn.ReLU(inplace=True),
            nn.Conv2d(384, 256, kernel_size=3, padding=1),          # output[192, 13, 13]
            nn.ReLU(inplace=True),
            nn.Conv2d(256, 256, kernel_size=3, padding=1),          # output[128, 13, 13]
            nn.ReLU(inplace=True),
            nn.MaxPool2d(kernel_size=3, stride=2),                  # output[128, 6, 6]
        )
        self.se_layer = SELayer(channel=256) 

        self.classifier = nn.Sequential(
            nn.Dropout(p=0.5),
            nn.Linear(256 * 6 * 6, 4096),
            # SELayer(channel=4096),
            nn.ReLU(inplace=True),
            nn.Dropout(p=0.5),
            nn.Linear(4096, 4096),
            nn.ReLU(inplace=True),
            nn.Linear(4096, num_classes),
        )
        if init_weights:
            self._initialize_weights()

    def forward(self, x):
        x = self.features(x)
        x = self.se_layer(x)
        x = torch.flatten(x, start_dim=1)
        x = self.classifier(x)
        return x

    def _initialize_weights(self):
        for m in self.modules():
            if isinstance(m, nn.Conv2d):
                nn.init.kaiming_normal_(m.weight, mode='fan_out', nonlinearity='relu')
                if m.bias is not None:
                    nn.init.constant_(m.bias, 0)
            elif isinstance(m, nn.Linear):
                nn.init.normal_(m.weight, 0, 0.01)
                nn.init.constant_(m.bias, 0)


class SELayer(nn.Module):
    def __init__(self, channel, reduction = 2):  # check point 1: 
        super(SELayer, self).__init__()
        self.avg_pool = nn.AdaptiveAvgPool2d(1)  # 全局平均池化，输入BCHW -> 输出 B*C*1*1
        self.fc = nn.Sequential(
            nn.Linear(channel, channel // reduction, bias=False),  # 可以看到channel得被reduction整除，否则可能出问题
            nn.ReLU(inplace=True),
            nn.Linear(channel // reduction, channel, bias=False),
            nn.Sigmoid()
        )

    def forward(self, x):
        b, c ,_ ,_= x.size()
        y = self.avg_pool(x).view(b, c)  # 得到B*C*1*1,然后转成B*C，才能送入到FC层中。
        y = self.fc(y).view(b, c, 1, 1)  # 得到B*C的向量，C个值就表示C个通道的权重。把B*C变为B*C*1*1是为了与四维的x运算。
        return x * y.expand_as(x)  # 先把B*C*1*1变成B*C*H*W大小，其中每个通道上的H*W个值都相等。*表示对应位置相乘
    

    #def forward(self, x):  
        # 应用全局平均池化，将x从[batch_size, channels, height, width]变为[batch_size, channels, 1, 1]  
        # 然后通过view操作将其变为[batch_size, channels]  
        #y = self.avg_pool(x).view(x.size(0), -1)  # 获取batch_size和channels，忽略其他维度  
        # 通过fc层得到每个通道的权重  
        #y = self.fc(y).view(x.size(0), x.size(1), 1, 1)  # 将权重重新塑形为[batch_size, channels, 1, 1]  
        # 将权重应用到原始输入x上，进行通道重标定  
        #return x * y.expand_as(x)