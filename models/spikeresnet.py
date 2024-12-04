import torch
import torch.nn as nn
from spikingjelly.activation_based import neuron, layer, functional

def conv3x3(in_planes, out_planes, stride=1, groups=1, dilation=1):
    """3x3 convolution with padding"""
    return layer.Conv2d(in_planes, out_planes, kernel_size=3, stride=stride,
                     padding=dilation, groups=groups, bias=False, dilation=dilation)


def conv1x1(in_planes, out_planes, stride=1):
    """1x1 convolution"""
    return layer.Conv2d(in_planes, out_planes, kernel_size=1, stride=stride, bias=False)


class BasicBlock(nn.Module):
    expansion = 1

    def __init__(self, inplanes, planes, stride=1, downsample=None, groups=1,
                 base_width=64, dilation=1, norm_layer=None, connect_f='ADD'):
        super(BasicBlock, self).__init__()
        self.connect_f = connect_f
        if norm_layer is None:
            norm_layer = layer.BatchNorm2d
        if groups != 1 or base_width != 64:
            raise ValueError('SpikingBasicBlock only supports groups=1 and base_width=64')
        if dilation > 1:
            raise NotImplementedError("Dilation > 1 not supported in SpikingBasicBlock")

        self.conv1 = nn.Sequential(
            conv3x3(inplanes, planes, stride),
            norm_layer(planes)
        )
        self.sn1 = neuron.LIFNode(detach_reset=True)
        self.conv2 = nn.Sequential(
            conv3x3(planes, planes),
            norm_layer(planes)
        )
        self.downsample = downsample
        self.stride = stride
        self.sn2 = neuron.LIFNode(detach_reset=True)

    def forward(self, x):
        identity = x

        out = self.sn1(self.conv1(x))

        out = self.sn2(self.conv2(out))

        if self.downsample is not None:
            identity = self.downsample(x)

        if self.connect_f == 'ADD':
            out += identity
        elif self.connect_f == 'AND':
            out *= identity
        elif self.connect_f == 'IAND':
            out = identity * (1. - out)
        else:
            raise NotImplementedError(self.connect_f)

        return out


class Bottleneck(nn.Module):
    expansion = 4
    def __init__(self, inplanes, planes, stride=1, downsample=None, groups=1,
                 base_width=64, dilation=1, norm_layer=None, connect_f='ADD'):
        super(Bottleneck, self).__init__()
        self.connect_f = connect_f
        if norm_layer is None:
            norm_layer = layer.BatchNorm2d
        width = int(planes * (base_width / 64.)) * groups
        self.conv1 = nn.Sequential(
            conv1x1(inplanes, width),
            norm_layer(width)
        )
        self.sn1 = neuron.LIFNode(detach_reset=True)

        self.conv2 = nn.Sequential(
            conv3x3(width, width, stride, groups, dilation),
            norm_layer(width)
        )
        self.sn2 = neuron.LIFNode(detach_reset=True)

        self.conv3 = nn.Sequential(
            conv1x1(width, planes * self.expansion),
            norm_layer(planes * self.expansion)
        )
        self.downsample = downsample
        self.stride = stride
        self.sn3 = neuron.LIFNode(detach_reset=True)

    def forward(self, x):
        identity = x

        out = self.sn1(self.conv1(x))

        out = self.sn2(self.conv2(out))

        out = self.sn3(self.conv3(out))

        if self.downsample is not None:
            identity = self.downsample(x)

        
        if self.connect_f == 'ADD':
            out += identity
        elif self.connect_f == 'AND':
            out *= identity
        elif self.connect_f == 'IAND':
            out = identity * (1. - out)
        else:
            raise NotImplementedError(self.connect_f)

        return out

class SEWResNet(nn.Module):
    def __init__(self, block, layers, num_classes=1000, C=3, H=48, W=48, T=4, 
                 zero_init_residual=False, groups=1, width_per_group=64, 
                 replace_stride_with_dilation=None, norm_layer=None, connect_f='ADD'):
        super(SEWResNet, self).__init__()
        self.T = T
        self.connect_f = connect_f
        if norm_layer is None:
            norm_layer = layer.BatchNorm2d
        self._norm_layer = norm_layer

        self.C = C
        self.H = H
        self.W = W

        self.inplanes = 64
        self.dilation = 1
        if replace_stride_with_dilation is None:
            replace_stride_with_dilation = [False, False, False]
        if len(replace_stride_with_dilation) != 3:
            raise ValueError("replace_stride_with_dilation should be None "
                             "or a 3-element tuple, got {}".format(replace_stride_with_dilation))
        self.groups = groups
        self.base_width = width_per_group

        self.conv1 = layer.Conv2d(self.C, self.inplanes, kernel_size=7, stride=2, padding=3,
                               bias=False)
        self.bn1 = layer.BatchNorm2d(self.inplanes)

        self.sn1 = neuron.LIFNode(detach_reset=True)
        self.maxpool = layer.MaxPool2d(kernel_size=3, stride=2, padding=1)

        self.layer1 = self._make_layer(block, 64, layers[0], connect_f=connect_f)
        self.layer2 = self._make_layer(block, 128, layers[1], stride=2,
                                       dilate=replace_stride_with_dilation[0], connect_f=connect_f)
        self.layer3 = self._make_layer(block, 256, layers[2], stride=2,
                                       dilate=replace_stride_with_dilation[1], connect_f=connect_f)
        self.layer4 = self._make_layer(block, 512, layers[3], stride=2,
                                       dilate=replace_stride_with_dilation[2], connect_f=connect_f)
        
        self.avgpool = layer.AdaptiveAvgPool2d((1, 1))

        self.classifier = nn.Linear(512 * block.expansion, num_classes)

        for m in self.modules():
            if isinstance(m, nn.Conv2d):
                nn.init.kaiming_normal_(m.weight, mode='fan_out', nonlinearity='relu')
            elif isinstance(m, (nn.BatchNorm2d, nn.GroupNorm)):
                nn.init.constant_(m.weight, 1)
                nn.init.constant_(m.bias, 0)

        functional.set_step_mode(self, 'm')

    def _make_layer(self, block, planes, blocks, stride=1, dilate=False, connect_f='ADD'):
        norm_layer = self._norm_layer
        downsample = None
        previous_dilation = self.dilation
        if dilate:
            self.dilation *= stride
            stride = 1
        if stride != 1 or self.inplanes != planes * block.expansion:
            downsample = nn.Sequential(
                conv1x1(self.inplanes, planes * block.expansion, stride),
                norm_layer(planes * block.expansion),
                neuron.LIFNode(detach_reset=True)
            )

        layers = []
        layers.append(block(self.inplanes, planes, stride, downsample, self.groups,
                            self.base_width, previous_dilation, norm_layer, connect_f))
        self.inplanes = planes * block.expansion
        for _ in range(1, blocks):
            layers.append(block(self.inplanes, planes, groups=self.groups,
                                base_width=self.base_width, dilation=self.dilation,
                                norm_layer=norm_layer, connect_f=connect_f))

        return nn.Sequential(*layers)

    def _forward_impl(self, x):
        feature_transform = 0.
        # (T, B, C, H, W)
        x = self.conv1(x)
        x = self.bn1(x)
        
        x = self.sn1(x)
        x = self.maxpool(x)
        x = self.layer1(x)
        x = self.layer2(x)
        x = self.layer3(x)
        x = self.layer4(x)

        x = self.avgpool(x)
        x = torch.flatten(x, 2)
        x = x.mean(0)

        return x, feature_transform

    def forward(self, x):
        functional.reset_net(self)

        if len(x.shape) == 5: # neuromorphic (B, T, C, H, W)
            x = x.transpose(0, 1) # ->(T, B, C, H, W)
        elif len(x.shape) == 4: # static (B, C, H, W)
            x = x.unsqueeze(0).repeat(self.T, 1, 1, 1, 1) # -> (T, B, C, H, W)
        else:
            raise NotImplementedError(f'Invalid inputs shape: {x.shape}')

        x, feature_transform = self._forward_impl(x)
        x = self.classifier(x) # (B, num_cls)

        return x, feature_transform


def _sew_resnet(block, layers, num_classes, C, H, W, T, **kwargs):
    model = SEWResNet(block, layers, num_classes, C, H, W, T, **kwargs)
    return model

def sew_resnet14(num_classes=100, C=3, H=32, W=32, T=4, **kwargs):
    return _sew_resnet(BasicBlock, [1, 2, 2, 1], num_classes, C, H, W, T, **kwargs)

def sew_resnet18(num_classes=100, C=3, H=32, W=32, T=4, **kwargs):
    return _sew_resnet(BasicBlock, [2, 2, 2, 2], num_classes, C, H, W, T, **kwargs)

def sew_resnet34(num_classes=100, C=3, H=32, W=32, T=4, **kwargs):
    return _sew_resnet(BasicBlock, [3, 4, 6, 3], num_classes, C, H, W, T, **kwargs)

def sew_resnet50(num_classes=100, C=3, H=32, W=32, T=4, **kwargs):
    return _sew_resnet(Bottleneck, [3, 4, 6, 3], num_classes, C, H, W, T, **kwargs)
