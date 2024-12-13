import os
import pickle
import random
import logging
import datetime
import numpy as np
from PIL import Image
from tqdm import tqdm

import torch
import torchvision
from torch.utils.data import Dataset
import torchvision.transforms as transforms

def get_local_time():
    cur = datetime.datetime.now()
    cur = cur.strftime('%b-%d-%Y_%H-%M-%S')
    return cur

class Logger:
    def __init__(self, name: str, log_file: str = "app.log", level: int = logging.INFO):
        ensure_dir(f'./logs')
        self.logger = logging.getLogger(name)
        self.logger.setLevel(level)
        self.logger.propagate = False
        self.logger.handlers.clear()

        self._add_file_handler(f'./logs/{log_file}', level)
        self._add_console_handler(level)

    def _add_file_handler(self, log_file: str, level: int):
        file_handler = logging.FileHandler(log_file, mode="w")
        file_handler.setLevel(level)
        file_handler.setFormatter(
            logging.Formatter("[%(asctime)s] - %(levelname)s - %(message)s")
        )
        self.logger.addHandler(file_handler)

    def _add_console_handler(self, level: int):
        console_handler = logging.StreamHandler()
        console_handler.setLevel(level)
        console_handler.setFormatter(
            logging.Formatter("%(message)s")
        )
        self.logger.addHandler(console_handler)

    def get_logger(self):
        return self.logger
    
def ensure_dir(path):
    if not os.path.exists(path):
        os.makedirs(path)

def seed_all(seed=1029):
    random.seed(seed)
    os.environ['PYTHONHASHSEED'] = str(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)  # if you are using multi-GPU.
    torch.backends.cudnn.benchmark = False
    torch.backends.cudnn.deterministic = True

class BaseDataset(Dataset):
    """Characterizes a dataset for PyTorch -- this dataset pre-loads all images in memory"""

    def __init__(self, data, transform, class_indices=None):
        """Initialization"""
        self.labels = data['y']
        self.images = data['x']
        self.transform = transform
        self.class_indices = class_indices

    def __len__(self):
        """Denotes the total number of samples"""
        return len(self.images)

    def __getitem__(self, index):
        """Generates one sample of data"""
        x = self.images[index] # (C, H, W) or str
        # static image nee transform, neuromorphic input have already been processed by spikingjelly
        if isinstance(x, str): # image root, load rgb image
            x = Image.open(x)
            x = x.convert("RGB")
            x = self.transform(x)
        else: # (C, H, W) numpy
            x = Image.fromarray(x)
            x = self.transform(x)

        y = self.labels[index]
        return x, y


class CaltechDataset(Dataset):
    def __init__(self, root='.', train=True, transform=None):
        super().__init__()

        dset = torchvision.datasets.Caltech101(root='.', download=True)
        self.data = []
        self.targets = []
        resize = transforms.Resize((224, 224), interpolation=transforms.InterpolationMode.BICUBIC)
        for x, y in tqdm(dset):
            x = x.convert("RGB")
            x = resize(x)
            x = np.array(x)

            self.data.append(x)
            self.targets.append(y)

        self.data = np.array(self.data)
        self.targets = np.array(self.targets)

        self.transform = transform
        
    def __len__(self):
        return len(self.targets)
    
    def __getitem__(self, index):
        x = self.data[index]
        if self.transform:
            x = Image.fromarray(x)
            x = self.transform(x)
        
        return x, self.targets[index]

class CUBDataset(Dataset):
    def __init__(self, root='./cub200', train=True, transform=None):
        super().__init__()

        if transform is None:
            self.transform = transforms.Compose([
                transforms.Resize(224, interpolation=transforms.InterpolationMode.BICUBIC),
                transforms.ToTensor(),
                transforms.Normalize(mean=(0.485, 0.456, 0.406), std=(0.229, 0.224, 0.225))
            ])
        else:
            self.transform = transform
        
        if train:
            dset_dir = os.path.join(root, 'train')
        else:
            dset_dir = os.path.join(root, 'test')
        dset = torchvision.datasets.ImageFolder(dset_dir)

        self.data, self.targets = self.split_images_labels(dset.imgs)

    def __len__(self):
        return len(self.targets)

    def __getitem__(self, index):
        img_dir = self.data[index]
        x = Image.open(img_dir)
        x = x.convert("RGB")
        if self.transform:
            x = self.transform(x)

        return x, self.targets[index]

    def split_images_labels(self, imgs):
        images, labels = [], []
        for item in imgs:
            images.append(item[0])
            labels.append(item[1])
        
        return np.array(images), np.array(labels)

class TinyImageNetDataset(Dataset):
    def __init__(self, root='./tiny-imagenet-200', train=True, transform=None):
        super().__init__()

        self.root = root
        self.transform = transform
        self.train = train

        if self.train:
            with open(os.path.join(self.root, 'train_dataset.pkl'), 'rb') as f:
                self.data, self.targets = pickle.load(f)
        else:
            with open(os.path.join(self.root, 'val_dataset.pkl'), 'rb') as f:
                self.data, self.targets = pickle.load(f)

        self.targets = self.targets.type(torch.LongTensor)

    def __getitem__(self, index):
        data = self.data[index].permute(1, 2, 0).numpy()
        data = Image.fromarray(data)
        if self.transform:
            data = self.transform(data)

        return data, self.targets[index] 
    
    def __len__(self):
        return len(self.targets)

class DVSCifar10(Dataset):
    def __init__(self, root, train=True, transform=None):
        super().__init__()

        self.transform = transform
        self.train = train

        data, targets = [], []
        self.root = os.path.join(root, 'train') if self.train else os.path.join(root, 'test')

        for f in os.listdir(self.root):
            d, t = torch.load(os.path.join(self.root, f)) # (C, H, W, T)
            d = d.permute([3, 0, 1, 2]) # (T, C, H, W)
            if self.transform:
                d = self.transform(d) # resize to new_H, new_w, (T, C, new_H, new_w)
            data.append(d) 
            targets.append(t)

        self.data = torch.stack(data, dim=0) # (B, T, C, H, W)
        self.targets = torch.cat(targets) # (B)
        self.targets = self.targets.long()

    def __getitem__(self, index):
        data = self.data[index]
        target = self.targets[index]

        return data, target
    
    def __len__(self):
        return len(self.targets) # len(os.listdir(self.root))

class NCaltech(Dataset):
    def __init__(self, data_path='./ncaltech/frames_number_10_split_by_time', data_type='train', transform=None):
        super().__init__()
        
        self.filepath = os.path.join(data_path)
        self.clslist = os.listdir(self.filepath)
        self.clslist.sort()

        self.data = [] # (B, T, C, H, W)
        self.targets = [] # (B)
        self.transform = transform

        for i, c in enumerate(self.clslist):
            print(i, c)
            file_list = os.listdir(os.path.join(self.filepath, c))
            num_file = len(file_list)
            cut_idx = int(num_file * 0.9) # 90% for train 10% for test
            train_file_list = file_list[:cut_idx]
            test_split_list = file_list[cut_idx:]

            for f in file_list:
                if data_type == 'train':
                    if f in train_file_list:
                        frame = np.load(os.path.join(self.filepath, c, f))['frames'] # (T, C, H, W)
                        frame = torch.from_numpy(frame).float()
                        if self.transform:
                            frame = self.transform(frame) # (T, C, new_H, new_W)

                        self.data.append(frame)
                        self.targets.append(i)
                else:
                    if f in test_split_list:
                        frame = np.load(os.path.join(self.filepath, c, f))['frames']
                        frame = torch.from_numpy(frame).float()
                        if self.transform:
                            frame = self.transform(frame)

                        self.data.append(frame)
                        self.targets.append(i)

        self.data = torch.stack(self.data, dim=0) # (B, T, C, H, W)
        self.targets = torch.tensor(self.targets) # (B)
        self.targets = self.targets.long()

    def __getitem__(self, index):
        data = self.data[index]
        label = self.targets[index]

        return data, label

    def __len__(self):
        return len(self.targets)
