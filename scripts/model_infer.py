import time
from tqdm import tqdm

import torch
import torch.nn as nn

import torchvision
import torchvision.transforms as transforms

from utils import *
from models.spikevgg import SpikeVGG9
from models.vit import VIT

dataset = 'imagenet'
model_name = 'vit'
device = 'cuda:1'

model_conf = {
    'svgg': SpikeVGG9,
    'vit': VIT,
}

if dataset == 'cifar100':
    transform_test = transforms.Compose([
        transforms.ToTensor(),
        transforms.Normalize((0.5071, 0.4867, 0.4408), (0.2675, 0.2565, 0.2761)),
    ])
    test_set = torchvision.datasets.CIFAR100(root= '.', train=False, download=False, transform=transform_test)

    num_classes = 100
    C, H, W = 3, 32, 32

elif dataset == 'imagenet':
    transform_test = transforms.Compose([
        transforms.Resize(224, interpolation=transforms.InterpolationMode.BICUBIC),
        transforms.ToTensor(),
        transforms.Normalize(mean=(0.485, 0.456, 0.406), std=(0.229, 0.224, 0.225))
    ])

    test_set = TinyImageNetDataset('./tiny-imagenet-200', train=False, transform=transform_test)

    num_classes = 200
    C, H, W = 3, 224, 224

test_loader = torch.utils.data.DataLoader(test_set, batch_size=1, shuffle=False)

cloud_model = model_conf[model_name](num_classes, C, H, W, 4, True)
cloud_model.load_state_dict(
    torch.load(f'saved/{dataset}/base0_task5/best_cloud_{model_name}.pt', map_location='cpu'))
cloud_model.eval()
cloud_model.to(device)
print('cloud model loaded successfully...')

total_time, total = 0, 0
with torch.no_grad():
    for images, targets in tqdm(test_loader, unit='batch'):
        # cloud infer
        start_time = time.time()
        cloud_outputs, _ = cloud_model(
            nn.functional.interpolate(images.to(device), size=(224, 224), mode='bilinear', align_corners=False)
        )
        total_time += (time.time() - start_time)
        total += 1

print(f'{model_name} inference lantency for {dataset} on server gpu: {total_time / total * 1000:.4f}ms')
