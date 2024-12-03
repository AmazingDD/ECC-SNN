import os
import pickle
import random
import logging
import datetime
import numpy as np
from PIL import Image

import torch
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
        x = Image.fromarray(self.images[index])
        x = self.transform(x)
        y = self.labels[index]
        return x, y

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

        self.root = root
        self.transform = transform
        self.train = train

        self.resize = transforms.Resize(size=(48, 48), interpolation=transforms.InterpolationMode.NEAREST)
        self.rotate = transforms.RandomRotation(degrees=30)
        self.shearx = transforms.RandomAffine(degrees=0, shear=(-30, 30))

    def __getitem__(self, index):
        data, target = torch.load(f'{self.root}/{index}.pt')
        data = self.resize(data.permute([3, 0, 1, 2]))

        if self.transform:
            choices = ['roll', 'rotate', 'shear']
            aug = np.random.choice(choices)

            if aug == 'roll':
                off1 = random.randint(-5, 5)
                off2 = random.randint(-5, 5)
                data = torch.roll(data, shifts=(off1, off2), dims=(2, 3))
            if aug == 'rotate':
                data = self.rotate(data)
            if aug == 'shear':
                data = self.shearx(data)

        return data, target.long().squeeze(-1)
    
    def __len__(self):
        return len(os.listdir(self.root))

class NCaltech(Dataset):
    def __init__(self, data_path='data/n-caltech/frames_number_10_split_by_number', data_type='train', transform=False):
        super().__init__()

        self.filepath = os.path.join(data_path)
        self.clslist = os.listdir(self.filepath)
        self.clslist.sort()

        self.dvs_filelist = []
        self.targets = []
        self.resize = transforms.Resize(size=(48, 48), interpolation=transforms.InterpolationMode.NEAREST)

        for i, c in enumerate(self.clslist):
            # print(i, c)
            file_list = os.listdir(os.path.join(self.filepath, c))
            num_file = len(file_list)

            cut_idx = int(num_file * 0.9)
            train_file_list = file_list[:cut_idx]
            test_split_list = file_list[cut_idx:]
            for f in file_list:
                if data_type == 'train':
                    if f in train_file_list:
                        self.dvs_filelist.append(os.path.join(self.filepath, c, f))
                        self.targets.append(i)
                    else:
                        if f in test_split_list:
                            self.dvs_filelist.append(os.path.join(self.filepath, c, f))
                            self.targets.append(i)
        
        self.data_num = len(self.dvs_filelist)
        self.data_type = data_type
        if data_type != 'train':
            counts = np.unique(np.array(self.targets), return_counts=True)[1]
            class_weights = counts.sum() / (counts * len(counts))
            self.class_weights = torch.Tensor(class_weights)

        self.classes = range(101)
        self.transform = transform
        self.rotate = transforms.RandomRotation(degrees=15)
        self.shearx = transforms.RandomAffine(degrees=0, shear=(-15, 15))

    def __getitem__(self, index):
        file_pth = self.dvs_filelist[index]
        label = self.targets[index]
        data = torch.from_numpy(np.load(file_pth)['frames']).float()
        data = self.resize(data)

        if self.transform:

            choices = ['roll', 'rotate', 'shear']
            aug = np.random.choice(choices)
            if aug == 'roll':
                off1 = random.randint(-3, 3)
                off2 = random.randint(-3, 3)
                data = torch.roll(data, shifts=(off1, off2), dims=(2, 3))
            if aug == 'rotate':
                data = self.rotate(data)
            if aug == 'shear':
                data = self.shearx(data)

        return data, label

    def __len__(self):
        return self.data_num
