import torch
import torch.nn as nn
from copy import deepcopy

class NetHead(nn.Module):
    def __init__(self, model):
        super().__init__()

        self.model = model

        last_layer = getattr(model, 'classifier')

        self.out_size = last_layer.in_features
        setattr(self.model, 'classifier', nn.Sequential()) # then logit turns to identity output from last layer

        self.heads = nn.ModuleList()
        self.task_cls = []
        self.task_offset = []

    def add_head(self, num_outputs):
        """Add a new head with the corresponding number of outputs. Also update the number of classes per task and the
        corresponding offsets
        """
        self.heads.append(nn.Linear(self.out_size, num_outputs))
        self.task_cls = torch.tensor([head.out_features for head in self.heads])
        self.task_offset = torch.cat([torch.LongTensor(1).zero_(), self.task_cls.cumsum(0)[:-1]])
    
    def forward(self, x):
        x = self.model(x) # None, feat_1, feat_2
        assert len(self.heads) > 0, 'Cannot access any head'
        y = []
        for head in self.heads:
            y.append(head(x[0]))

        return y, x[1]
    
    def freeze_bn(self):
        """Freeze all Batch Normalization layers from the model and use them in eval() mode"""
        for m in self.model.modules():
            if isinstance(m, nn.BatchNorm2d):
                m.eval()

    def freeze_backbone(self):
        """Freeze all parameters from the main model, but not the heads"""
        for param in self.model.parameters():
            param.requires_grad = False

    def freeze_all(self):
        """Freeze all parameters from the model, including the heads"""
        for param in self.parameters():
            param.requires_grad = False

    def set_state_dict(self, state_dict):
        """Load weights into the model"""
        self.load_state_dict(deepcopy(state_dict))
        return
    
    def get_copy(self):
        """Get weights from the model"""
        return deepcopy(self.state_dict())

