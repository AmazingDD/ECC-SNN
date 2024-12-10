import torch
import torch.nn as nn

from timm.models import create_model

class VGG16(nn.Module):
    def __init__(self, num_classes=10, C=3, H=32, W=32, T=4, pretrain=False):
        super(VGG16, self).__init__()

        self.pretrain = pretrain

        if pretrain:
            try:
                self.vgg = create_model('vgg16.tv_in1k', pretrained=True)
            except:
                print('Fail to load model from HF hub')
                self.vgg = create_model('vgg16.tv_in1k', pretrained=False)
                self.vgg.load_state_dict(torch.load('./vgg16.pt', map_location='cpu'))

            for param in self.vgg.parameters():
                param.requires_grad = False

        else:
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

        self.classifier = nn.Linear(4096, num_classes)

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
        feature_transform = None
        if self.pretrain:
            for i, layer in enumerate(self.vgg.features):
                x = layer(x)
                if i == 16:
                    feature_transform = x.detach()

            x = self.vgg.pre_logits(x)
            x = self.vgg.head.global_pool(x)
            x = self.vgg.head.drop(x)
            x = x.view(x.size(0), -1)
        else:
            # make sure this input only have dimension (B, C, H, W)
            for i, layer in enumerate(self.features):
                x = layer(x)
                if i == 23: # rough index for vgg9 student
                    feature_transform = x.detach() # [B, 256, H//8, W//8]

            x = x.view(x.size(0), -1)
            x = self.mlp(x)

        logit = self.classifier(x)

        feature_transform = nn.functional.adaptive_avg_pool2d(feature_transform, (1, 1)) # [B, 256, 1, 1]

        return logit, feature_transform 
