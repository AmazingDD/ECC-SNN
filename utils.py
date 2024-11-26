import os
import random
import logging
import numpy as np
from PIL import Image

import torch
from torch.utils.data import Dataset
import torchvision.transforms as transforms

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
    def __init__(self, data_dir='./tiny-imagenet-200', transform=None, split='train'):
        super().__init__()

        self.data_dir = data_dir
        self.transform = transform
        self.split = split
        self.images = []
        self.labels = []

        if split == 'train':
            self._load_train_data()
        elif split == 'val':
            self._load_val_data()
        elif split == 'test':
            self._load_test_data()

    def _load_train_data(self):
        for class_name in os.listdir(os.path.join(self.data_dir, 'train')):
            class_dir = os.path.join(self.data_dir, 'train', class_name, 'images')
            if os.path.isdir(class_dir):
                for img_name in os.listdir(class_dir):
                    self.images.append(os.path.join(class_dir, img_name))
                    self.labels.append(class_name)

    def _load_val_data(self):
        val_dir = os.path.join(self.data_dir, 'val')
        annotations_path = os.path.join(val_dir, 'val_annotations.txt')
        with open(annotations_path, 'r') as f:
            annotations = f.readlines()
        img_to_class = {line.split('\t')[0]: line.split('\t')[1] for line in annotations}
        for img_name, class_name in img_to_class.items():
            self.images.append(os.path.join(val_dir, 'images', img_name))
            self.labels.append(class_name)

    def _load_test_data(self):
        test_dir = os.path.join(self.data_dir, 'test', 'images')
        for img_name in os.listdir(test_dir):
            self.images.append(os.path.join(test_dir, img_name))
        self.labels = [-1] * len(self.images)

    def __len__(self):
        return len(self.images)
    
    def __getitem__(self, idx):
        img_path = self.images[idx]
        img = Image.open(img_path).convert('RGB')
        if self.transform:
            img = self.transform(img)
        label = self.labels[idx]
        return img, label
    
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

class NCaltech101(Dataset):
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
