import torch
import torch.nn as nn

from timm.models import create_model

class BasicBlock(nn.Module):
    """Basic Block for resnet 18 and resnet 34

    """
    expansion = 1

    def __init__(self, in_channels, out_channels, stride=1):
        super().__init__()

        #residual function
        self.residual_function = nn.Sequential(
            nn.Conv2d(in_channels, out_channels, kernel_size=3, stride=stride, padding=1, bias=False),
            nn.BatchNorm2d(out_channels),
            nn.ReLU(inplace=True),
            nn.Conv2d(out_channels, out_channels * BasicBlock.expansion, kernel_size=3, padding=1, bias=False),
            nn.BatchNorm2d(out_channels * BasicBlock.expansion)
        )

        #shortcut
        self.shortcut = nn.Sequential()

        #the shortcut output dimension is not the same with residual function
        #use 1*1 convolution to match the dimension
        if stride != 1 or in_channels != BasicBlock.expansion * out_channels:
            self.shortcut = nn.Sequential(
                nn.Conv2d(in_channels, out_channels * BasicBlock.expansion, kernel_size=1, stride=stride, bias=False),
                nn.BatchNorm2d(out_channels * BasicBlock.expansion)
            )

    def forward(self, x):
        return nn.ReLU(inplace=True)(self.residual_function(x) + self.shortcut(x))

class BottleNeck(nn.Module):
    """Residual block for resnet over 50 layers

    """
    expansion = 4
    def __init__(self, in_channels, out_channels, stride=1):
        super().__init__()
        self.residual_function = nn.Sequential(
            nn.Conv2d(in_channels, out_channels, kernel_size=1, bias=False),
            nn.BatchNorm2d(out_channels),
            nn.ReLU(inplace=True),
            nn.Conv2d(out_channels, out_channels, stride=stride, kernel_size=3, padding=1, bias=False),
            nn.BatchNorm2d(out_channels),
            nn.ReLU(inplace=True),
            nn.Conv2d(out_channels, out_channels * BottleNeck.expansion, kernel_size=1, bias=False),
            nn.BatchNorm2d(out_channels * BottleNeck.expansion),
        )

        self.shortcut = nn.Sequential()

        if stride != 1 or in_channels != out_channels * BottleNeck.expansion:
            self.shortcut = nn.Sequential(
                nn.Conv2d(in_channels, out_channels * BottleNeck.expansion, stride=stride, kernel_size=1, bias=False),
                nn.BatchNorm2d(out_channels * BottleNeck.expansion)
            )

    def forward(self, x):
        return nn.ReLU(inplace=True)(self.residual_function(x) + self.shortcut(x))

class ResNet(nn.Module):

    def __init__(self, block, num_block, num_classes=100, C=3, H=32, W=32, T=4):
        super().__init__()

        self.C = C
        self.num_classes = num_classes
        self.T = T
        self.H = H
        self.W = W

        self.in_channels = 64

        self.conv1 = nn.Sequential(
            nn.Conv2d(self.C, 64, kernel_size=3, padding=1, bias=False),
            nn.BatchNorm2d(64),
            nn.ReLU(inplace=True))
        #we use a different inputsize than the original paper
        #so conv2_x's stride is 1
        self.conv2_x = self._make_layer(block, 64, num_block[0], 1)
        self.conv3_x = self._make_layer(block, 128, num_block[1], 2)
        self.conv4_x = self._make_layer(block, 256, num_block[2], 2)
        self.conv5_x = self._make_layer(block, 512, num_block[3], 2)
        self.avg_pool = nn.AdaptiveAvgPool2d((1, 1))

        self.classifier = nn.Linear(512 * block.expansion, self.num_classes)

    def _make_layer(self, block, out_channels, num_blocks, stride):
        """make resnet layers(by layer i didnt mean this 'layer' was the
        same as a neuron netowork layer, ex. conv layer), one layer may
        contain more than one residual block

        Args:
            block: block type, basic block or bottle neck block
            out_channels: output depth channel number of this layer
            num_blocks: how many blocks per layer
            stride: the stride of the first block of this layer

        Return:
            return a resnet layer
        """

        # we have num_block blocks per layer, the first block
        # could be 1 or 2, other blocks would always be 1
        strides = [stride] + [1] * (num_blocks - 1)
        layers = []
        for stride in strides:
            layers.append(block(self.in_channels, out_channels, stride))
            self.in_channels = out_channels * block.expansion

        return nn.Sequential(*layers)

    def _forward(self, x):
        output = self.conv1(x)
        output = self.conv2_x(output)
        output = self.conv3_x(output)
        output = self.conv4_x(output)
        output = self.conv5_x(output)
        output = self.avg_pool(output)
        output = output.view(output.size(0), -1)

        output = self.classifier(output)

        return output
    
    def forward(self, x):
        logit = 0.
        if len(x.shape) == 5: # neuromorphic (B, T, C, H, W)
            for ts in range(self.T):
                tmp = self._forward(x[:, ts, ...]) 
                logit += tmp
            logit /= self.T
        elif len(x.shape) == 4: # static (B, C, H, W)
            logit = self._forward(x)
        else:
            raise NotImplementedError(f'Invalid inputs shape: {x.shape}')

        self._forward(x)

        # resnet only for cloud model in our study, the SNN-based resnet perform not very well with training from scratch
        return logit, None 

class ResNet34(nn.Module):  
    def __init__(self, num_classes=100, C=3, H=32, W=32, T=4):
        super().__init__()

        try:
            self.features = create_model('resnet34.a1_in1k', pretrained=True)
        except:
            print('Fail to load model from HF hub')
            self.features = create_model('resnet34.a1_in1k', pretrained=False)
            self.features.load_state_dict(torch.load('./resnet34.pt', map_location='cpu'))

        for param in self.features.parameters():
            param.requires_grad = False

        self.classifier = nn.Linear(self.features.fc.in_features, num_classes)
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
        feature_transform = 0.

        features = self.features.forward_features(x) 
        
        features = nn.functional.adaptive_avg_pool2d(features, (1, 1)) 
        features = features.view(features.size(0), -1) 

        logit = self.classifier(features) # (B, cls)

        return logit, feature_transform 


class ResNet50(nn.Module):
    def __init__(self, num_classes=100, C=3, H=32, W=32, T=4):
        super().__init__()

        try:
            self.features = create_model('resnet50', pretrained=True)
        except:
            print('Fail to load model from HF hub')
            self.features = create_model('resnet50', pretrained=False)
            self.features.load_state_dict(torch.load('./resnet50.pt', map_location='cpu'))

        for param in self.features.parameters():
            param.requires_grad = False

        self.classifier = nn.Linear(self.features.fc.in_features, num_classes)
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
        feature_transform = 0.
        
        features = self.features.forward_features(x) # (B, 2048, 7, 7)
        
        features = nn.functional.adaptive_avg_pool2d(features, (1, 1)) # (B, 2048, 1, 1)
        features = features.view(features.size(0), -1) 

        logit = self.classifier(features) # (B, cls)

        return logit, feature_transform 
    
def resnet14(num_classes=100, C=3, H=32, W=32, T=4):
    return ResNet(BasicBlock, [1, 2, 2, 1], num_classes, C, H, W, T)

def resnet18(num_classes=100, C=3, H=32, W=32, T=4):
    return ResNet(BasicBlock, [2, 2, 2, 2], num_classes, C, H, W, T)

def resnet34(num_classes=100, C=3, H=32, W=32, T=4, pretrain=False):
    if pretrain:
        return ResNet34(num_classes, C, H, W, T)
    else:
        return ResNet(BasicBlock, [3, 4, 6, 3], num_classes, C, H, W, T)

def resnet50(num_classes=100, C=3, H=32, W=32, T=4, pretrain=False):
    if pretrain:
        return ResNet50(num_classes, C, H, W, T)
    else:
        return ResNet(BottleNeck, [3, 4, 6, 3], num_classes, C, H, W, T)
