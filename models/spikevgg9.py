import torch.nn as nn
from spikingjelly.activation_based import layer, functional, neuron, surrogate

class Transform(nn.Module):
    def __init__(self, dim=256) -> None:
        super().__init__()
        self.conv = layer.Conv2d(in_channels=dim, out_channels=dim, kernel_size=1, stride=1)
        self.bn = layer.BatchNorm2d(dim)
        
    def forward(self, x):
        return self.bn(self.conv(x))

class SpikeVGG9(nn.Module):
    def __init__(self, num_classes, C=3, H=32, W=32, T=4):
        super(SpikeVGG9, self).__init__()
        self.features = nn.Sequential(
            layer.Conv2d(C, 64, kernel_size=(3, 3), stride=(1, 1), padding=(1, 1), bias=False),
            layer.BatchNorm2d(64, eps=1e-5, momentum=0.1, affine=True, track_running_stats=True),
            neuron.LIFNode(v_threshold=1.0, v_reset=0.0, detach_reset=True, tau=2.0),
            layer.Conv2d(64, 64, kernel_size=(3, 3), stride=(1, 1), padding=(1, 1), bias=False),
            layer.BatchNorm2d(64, eps=1e-5, momentum=0.1, affine=True, track_running_stats=True),
            neuron.LIFNode(v_threshold=1.0, v_reset=0.0, detach_reset=True, tau=2.0),
            layer.MaxPool2d(kernel_size=2, stride=2, padding=0),
            layer.Conv2d(64, 128, kernel_size=(3, 3), stride=(1, 1), padding=(1, 1), bias=False),
            layer.BatchNorm2d(128, eps=1e-5, momentum=0.1, affine=True, track_running_stats=True),
            neuron.LIFNode(v_threshold=1.0, v_reset=0.0, detach_reset=True, tau=2.0),
            layer.Conv2d(128, 128, kernel_size=(3, 3), stride=(1, 1), padding=(1, 1), bias=False),
            layer.BatchNorm2d(128, eps=1e-5, momentum=0.1, affine=True, track_running_stats=True),
            neuron.LIFNode(v_threshold=1.0, v_reset=0.0, detach_reset=True, tau=2.0),
            layer.MaxPool2d(kernel_size=2, stride=2, padding=0),
            layer.Conv2d(128, 256, kernel_size=(3, 3), stride=(1, 1), padding=(1, 1), bias=False),
            layer.BatchNorm2d(256, eps=1e-5, momentum=0.1, affine=True, track_running_stats=True),
            neuron.LIFNode(v_threshold=1.0, v_reset=0.0, detach_reset=True, tau=2.0),
            layer.Conv2d(256, 256, kernel_size=(3, 3), stride=(1, 1), padding=(1, 1), bias=False),
            layer.BatchNorm2d(256, eps=1e-5, momentum=0.1, affine=True, track_running_stats=True),
            neuron.LIFNode(v_threshold=1.0, v_reset=0.0, detach_reset=True, tau=2.0),
            layer.Conv2d(256, 256, kernel_size=(3, 3), stride=(1, 1), padding=(1, 1), bias=False),
            layer.BatchNorm2d(256, eps=1e-5, momentum=0.1, affine=True, track_running_stats=True),
            neuron.LIFNode(v_threshold=1.0, v_reset=0.0, detach_reset=True, tau=2.0),
            layer.MaxPool2d(kernel_size=2, stride=2, padding=0),
        )

        self.transform = Transform(dim=256)

        self.mlp = nn.Sequential(
            layer.Flatten(start_dim=1, end_dim=-1),
            layer.Linear(in_features=256 * (H // 8) * (W // 8), out_features=1024, bias=False),
            neuron.LIFNode(v_threshold=1.0, v_reset=0.0, detach_reset=True, tau=2.0),
        )
        
        self.classifier = nn.Linear(in_features=1024, out_features=num_classes, bias=False)

        self.T = T
        functional.set_step_mode(self, "m") 

    def forward(self, x):
        functional.reset_net(self)

        if len(x.shape) == 5: # neuromorphic (B, T, C, H, W)
            x = x.transpose(0, 1) # ->(T, B, C, H, W)
        elif len(x.shape) == 4: # static (B, C, H, W)
            x = x.unsqueeze(0).repeat(self.T, 1, 1, 1, 1) # -> (T, B, C, H, W)
        else:
            raise NotImplementedError(f'Invalid inputs shape: {x.shape}')
            
        x = self.features(x) #　[T, B, 256, 4, 4]
        feature_transform = self.transform(x).mean(0).detach() # [B, 256, 4, 4]

        x = self.mlp(x).mean(0)

        logit = self.classifier(x)

        return logit, feature_transform
