import torch
import torch.nn as nn

from timm.models import create_model

class VIT(nn.Module):
    def __init__(self, num_classes=200, C=3, H=224, W=224, T=4, pretrain=True):
        super().__init__()

        if pretrain:
            try:
                self.vit = create_model('vit_base_patch16_224', pretrained=True, drop_path_rate=0.1)
            except:
                print('Fail to load model from HF hub')
                self.vit = create_model('vit_base_patch16_224', pretrained=False, drop_path_rate=0.1)
                self.vit.load_state_dict(torch.load('./vit_base_patch16_224.pt', map_location='cpu'))

            for param in self.vit.parameters():
                param.requires_grad = False
        else:
            self.vit = create_model('vit_base_patch16_224', pretrained=False, drop_path_rate=0.1)

            self.vit.patch_embed.img_size = (H, W)
            patch_size = 16
            self.vit.patch_embed.proj = nn.Conv2d(C, self.vit.embed_dim, kernel_size=(patch_size, patch_size), stride=(patch_size, patch_size), padding=0)
            num_patches = (H // patch_size) * (W // patch_size)
            self.vit.patch_embed.num_patches = num_patches
            self.vit.pos_embed = nn.Parameter(torch.zeros(1, num_patches + 1, self.vit.embed_dim))


        self.classifier = nn.Linear(self.vit.embed_dim, num_classes)
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
        
        features = self.vit.forward_features(x) # (B, patch_num + 1, D)
        cls_token_features = features[:, 0, :] # (B, D)

        logit = self.classifier(cls_token_features) # (B, cls)

        return logit, feature_transform 
