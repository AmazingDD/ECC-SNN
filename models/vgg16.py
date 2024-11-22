import torch.nn as nn

class VGG16(nn.Module):
    def __init__(self, num_classes=10, C=3, H=32, W=32, T=4):
        super(VGG16, self).__init__()
        self.features = nn.Sequential(
            nn.Conv2d(C, 64, kernel_size=3, padding=1),
            nn.BatchNorm2d(64),
            nn.ReLU(inplace=True),
            nn.Conv2d(64, 64, kernel_size=3, padding=1),
            nn.BatchNorm2d(64),
            nn.ReLU(inplace=True),
            nn.MaxPool2d(kernel_size=2, stride=2),

            nn.Conv2d(64, 128, kernel_size=3, padding=1),
            nn.BatchNorm2d(128),
            nn.ReLU(inplace=True),
            nn.Conv2d(128, 128, kernel_size=3, padding=1),
            nn.BatchNorm2d(128),
            nn.ReLU(inplace=True),
            nn.MaxPool2d(kernel_size=2, stride=2),

            nn.Conv2d(128, 256, kernel_size=3, padding=1),
            nn.BatchNorm2d(256),
            nn.ReLU(inplace=True),
            nn.Conv2d(256, 256, kernel_size=3, padding=1),
            nn.BatchNorm2d(256),
            nn.ReLU(inplace=True),
            nn.Conv2d(256, 256, kernel_size=3, padding=1),
            nn.BatchNorm2d(256),
            nn.ReLU(inplace=True),  
            nn.MaxPool2d(kernel_size=2, stride=2), # 23, this is the overlapping endpoint of vgg9 and vgg16

            nn.Conv2d(256, 512, kernel_size=3, padding=1),
            nn.BatchNorm2d(512),
            nn.ReLU(inplace=True),
            nn.Conv2d(512, 512, kernel_size=3, padding=1),
            nn.BatchNorm2d(512),
            nn.ReLU(inplace=True),
            nn.Conv2d(512, 512, kernel_size=3, padding=1),
            nn.BatchNorm2d(512),
            nn.ReLU(inplace=True),
            nn.MaxPool2d(kernel_size=2, stride=2),

            nn.Conv2d(512, 512, kernel_size=3, padding=1),
            nn.BatchNorm2d(512),
            nn.ReLU(inplace=True),
            nn.Conv2d(512, 512, kernel_size=3, padding=1),
            nn.BatchNorm2d(512),
            nn.ReLU(inplace=True),
            nn.Conv2d(512, 512, kernel_size=3, padding=1),
            nn.BatchNorm2d(512),
            nn.ReLU(inplace=True),
            nn.MaxPool2d(kernel_size=2, stride=2),
        )

        self.mlp = nn.Sequential(
            nn.Linear(512 * (H // 32) * (W // 32), 4096),  # 5 maxpool, so divided by 32
            nn.ReLU(inplace=True),
            nn.Dropout(),
            nn.Linear(4096, 4096),
            nn.ReLU(inplace=True),
            nn.Dropout(),
        )

        self.classifier_head = nn.Linear(4096, num_classes)

        self.T = T

    def forward(self, x):  
        logit, feature_transform = 0., 0.
        if len(x.shape) == 5: # neuromorphic (B, T, C, H, W)
            for ts in range(self.T):
                a, b = self._forward(x[:, ts, ...]) 
                logit += a
                feature_transform += b

            logit /= self.T
            feature_transform /= self.T
        elif len(x.shape) == 4: # static (B, C, H, W)
            logit, feature_transform = self._forward(x)
        else:
            raise NotImplementedError(f'Invalid inputs shape: {x.shape}')
        
        return logit, feature_transform

    def _forward(self, x):
        # make sure this input only have dimension (B, C, H, W)
        feature_transform = None
        for i, layer in enumerate(self.features):
            x = layer(x)
            if i == 23: # rough index for vgg9 student
                feature_transform = x.detach() # [B, 256, H//8, W//8]

        x = x.view(x.size(0), -1)
        x = self.mlp(x)

        logit = self.classifier_head(x)

        return logit, feature_transform 
