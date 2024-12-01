import torch.nn as nn

from spikingjelly.activation_based import neuron, layer, functional

class MLP(nn.Module):
    def __init__(self, in_features, hidden_features=None, out_features=None):
        super().__init__()
        out_features = out_features or in_features
        hidden_features = hidden_features or in_features
        self.fc1_linear = layer.Linear(in_features, hidden_features)
        self.fc1_bn = layer.BatchNorm1d(hidden_features)
        self.fc1_lif = neuron.LIFNode(tau=2.0, detach_reset=True)

        self.fc2_linear = layer.Linear(hidden_features, out_features)
        self.fc2_bn = layer.BatchNorm1d(out_features)
        self.fc2_lif = neuron.LIFNode(tau=2.0, detach_reset=True)

        self.c_hidden = hidden_features

    def forward(self, x):
        T, B, N, C = x.shape

        x = self.fc1_linear(x)
        x = self.fc1_bn(x.transpose(-1, -2)).transpose(-1, -2).reshape(T, B, N, self.c_hidden).contiguous()
        x = self.fc1_lif(x)

        x = self.fc2_linear(x)
        x = self.fc2_bn(x.transpose(-1, -2)).transpose(-1, -2).reshape(T, B, N, C).contiguous()
        x = self.fc2_lif(x)

        return x
    
class SSA(nn.Module):
    def __init__(self, dim, num_heads=8):
        super().__init__()
        assert dim % num_heads == 0, f"dim {dim} should be divided by num_heads {num_heads}."
        self.dim = dim
        self.num_heads = num_heads

        self.scale = 0.125
        
        self.q_linear = layer.Linear(dim, dim)
        self.q_bn = layer.BatchNorm1d(dim)
        self.q_lif = neuron.LIFNode(tau=2.0, detach_reset=True)

        self.k_linear = layer.Linear(dim, dim)
        self.k_bn = layer.BatchNorm1d(dim)
        self.k_lif = neuron.LIFNode(tau=2.0, detach_reset=True)

        self.v_linear = layer.Linear(dim, dim)
        self.v_bn = layer.BatchNorm1d(dim)
        self.v_lif = neuron.LIFNode(tau=2.0, detach_reset=True)
        self.attn_lif = neuron.LIFNode(tau=2.0, v_threshold=0.5, detach_reset=True)

        self.proj_linear = layer.Linear(dim, dim)
        self.proj_bn = layer.BatchNorm1d(dim)
        self.proj_lif = neuron.LIFNode(tau=2.0, detach_reset=True)

    def forward(self, x):
        T, B, N, C = x.shape

        x_for_qkv = x.clone()

        q_linear_out = self.q_linear(x_for_qkv)  
        q_linear_out = self.q_bn(q_linear_out.transpose(-1, -2)).transpose(-1, -2).reshape(T, B, N, C).contiguous()
        q_linear_out = self.q_lif(q_linear_out)
        q = q_linear_out.reshape(T, B, N, self.num_heads, C // self.num_heads).permute(0, 1, 3, 2, 4).contiguous()

        k_linear_out = self.k_linear(x_for_qkv)
        k_linear_out = self.k_bn(k_linear_out.transpose(-1, -2)).transpose(-1, -2).reshape(T, B, N, C).contiguous()
        k_linear_out = self.k_lif(k_linear_out)
        k = k_linear_out.reshape(T, B, N, self.num_heads, C // self.num_heads).permute(0, 1, 3, 2, 4).contiguous()

        v_linear_out = self.v_linear(x_for_qkv)
        v_linear_out = self.v_bn(v_linear_out.transpose(-1, -2)).transpose(-1, -2).reshape(T, B, N, C).contiguous()
        v_linear_out = self.v_lif(v_linear_out)
        v = v_linear_out.reshape(T, B, N, self.num_heads, C // self.num_heads).permute(0, 1, 3, 2, 4).contiguous()

        attn = (q @ k.transpose(-2, -1)) * self.scale
        x = attn @ v
        x = x.transpose(2, 3).reshape(T, B, N, C).contiguous()
        x = self.attn_lif(x)
        x = self.proj_lif(self.proj_bn(self.proj_linear(x).transpose(-1, -2)).transpose(-1, -2).reshape(T, B, N, C))

        return x
    
class Block(nn.Module):
    def __init__(self, dim, num_heads, mlp_ratio=4.):
        super().__init__()

        self.attn = SSA(dim, num_heads=num_heads)
        mlp_hidden_dim = int(dim * mlp_ratio)
        self.mlp = MLP(in_features=dim, hidden_features=mlp_hidden_dim)

    def forward(self, x):
        x = x + self.attn(x)
        x = x + self.mlp(x)
        return x
    
class SPS(nn.Module):
    def __init__(self, in_channels=2, img_size_h=128, img_size_w=128, patch_size=4, embed_dims=256):
        super().__init__()
        self.image_size = [img_size_h, img_size_w]

        patch_size = (patch_size, patch_size)
        self.patch_size = patch_size
        self.C = in_channels
        self.H, self.W = self.image_size[0] // patch_size[0], self.image_size[1] // patch_size[1]

        self.num_patches = self.H * self.W
        self.proj_conv = layer.Conv2d(in_channels, embed_dims // 8, kernel_size=3, stride=1, padding=1, bias=False)
        self.proj_bn = layer.BatchNorm2d(embed_dims // 8)
        self.proj_lif = neuron.LIFNode(tau=2.0, detach_reset=True)

        self.proj_conv1 = layer.Conv2d(embed_dims // 8, embed_dims // 4, kernel_size=3, stride=1, padding=1, bias=False)
        self.proj_bn1 = layer.BatchNorm2d(embed_dims // 4)
        self.proj_lif1 = neuron.LIFNode(tau=2.0, detach_reset=True)

        self.proj_conv2 = layer.Conv2d(embed_dims//4, embed_dims//2, kernel_size=3, stride=1, padding=1, bias=False)
        self.proj_bn2 = layer.BatchNorm2d(embed_dims//2)
        self.proj_lif2 = neuron.LIFNode(tau=2.0, detach_reset=True)
        self.maxpool2 = layer.MaxPool2d(kernel_size=3, stride=2, padding=1, dilation=1, ceil_mode=False)

        self.proj_conv3 = layer.Conv2d(embed_dims//2, embed_dims, kernel_size=3, stride=1, padding=1, bias=False)
        self.proj_bn3 = layer.BatchNorm2d(embed_dims)
        self.proj_lif3 = neuron.LIFNode(tau=2.0, detach_reset=True)
        self.maxpool3 = layer.MaxPool2d(kernel_size=3, stride=2, padding=1, dilation=1, ceil_mode=False)

        self.rpe_conv = layer.Conv2d(embed_dims, embed_dims, kernel_size=3, stride=1, padding=1, bias=False)
        self.rpe_bn = layer.BatchNorm2d(embed_dims)
        self.rpe_lif = neuron.LIFNode(tau=2.0, detach_reset=True)

    def forward(self, x):
        T, B, C, H, W = x.shape
        x = self.proj_conv(x)
        x = self.proj_bn(x).reshape(T, B, -1, H, W).contiguous()
        x = self.proj_lif(x).contiguous()

        x = self.proj_conv1(x)
        x = self.proj_bn1(x).reshape(T, B, -1, H, W).contiguous()
        x = self.proj_lif1(x).contiguous()

        x = self.proj_conv2(x)
        x = self.proj_bn2(x).reshape(T, B, -1, H, W).contiguous()
        x = self.proj_lif2(x).contiguous()
        x = self.maxpool2(x)

        x = self.proj_conv3(x)
        x = self.proj_bn3(x).reshape(T, B, -1, H // 2, W // 2).contiguous()
        x = self.proj_lif3(x).contiguous()
        x = self.maxpool3(x)

        x_feat = x.reshape(T, B, -1, H // 4, W // 4).contiguous()

        x = self.rpe_conv(x)
        x = self.rpe_bn(x).reshape(T, B, -1, H // 4, W // 4).contiguous()
        x = self.rpe_lif(x)
        x = x + x_feat

        x = x.flatten(-2).transpose(-1, -2)  # T,B,N,C

        return x
    
class Transform(nn.Module):
    def __init__(self, dim) -> None:
        super().__init__()
        self.linear = layer.Linear(dim, dim)
        self.bn = layer.BatchNorm1d(512)
        
    def forward(self, x):
        # T, B, N, C
        return self.bn(self.linear(x.transpose(-1, -2))).transpose(-1, -2).contiguous()


class SVIT(nn.Module):
    def __init__(self, num_classes=10, C=2, H=128, W=128, T=4):
        super().__init__()

        self.T = T
        self.num_classes = num_classes

        self.depths = 4
        self.embed_dim = 512
        self.patch_size = 16

        self.patch_embed = SPS(
            in_channels=C,
            img_size_h=H,
            img_size_w=W,
            patch_size=self.patch_size,
            embed_dims=self.embed_dim)
        
        self.block = nn.ModuleList([Block(dim=self.embed_dim, num_heads=8, mlp_ratio=4) for _ in range(self.depths)])

        self.classifier = nn.Linear(self.embed_dim, num_classes) if num_classes > 0 else nn.Identity()

        self.apply(self._init_weights)

        functional.set_step_mode(self, "m") 

    def _init_weights(self, m):
        if isinstance(m, nn.Linear):
            nn.init.trunc_normal_(m.weight, std=.02)
            if isinstance(m, nn.Linear) and m.bias is not None:
                nn.init.constant_(m.bias, 0)

    def forward_features(self, x):
        block = getattr(self, f"block")
        patch_embed = getattr(self, f"patch_embed")

        feature_transform = 0.

        x = patch_embed(x)
    
        for blk in block:
            x = blk(x)
                
        return x.mean(2), feature_transform

    def forward(self, x):
        functional.reset_net(self)

        if len(x.shape) == 5: # neuromorphic (B, T, C, H, W)
            x = x.transpose(0, 1) # ->(T, B, C, H, W)
        elif len(x.shape) == 4: # static (B, C, H, W)
            x = x.unsqueeze(0).repeat(self.T, 1, 1, 1, 1) # -> (T, B, C, H, W)
        else:
            raise NotImplementedError(f'Invalid inputs shape: {x.shape}')

        x, feature_transform = self.forward_features(x)
        x = x.mean(0)
        
        x = self.classifier(x)

        return x, feature_transform
