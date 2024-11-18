import os
import random
import logging
import numpy as np
from PIL import Image

import torch
from torch.utils.data import Dataset

class Logger:
    def __init__(self, name: str, log_file: str = "app.log", level: int = logging.INFO):
        ensure_dir(f'logs/{log_file}')
        self.logger = logging.getLogger(name)
        self.logger.setLevel(level)
        self._add_file_handler(log_file, level)
        self._add_console_handler(level)

    def _add_file_handler(self, log_file: str, level: int):
        file_handler = logging.FileHandler(log_file, mode="a")
        file_handler.setLevel(level)
        file_handler.setFormatter(
            logging.Formatter("%(asctime)s - %(name)s - %(levelname)s - %(message)s")
        )
        self.logger.addHandler(file_handler)

    def _add_console_handler(self, level: int):
        console_handler = logging.StreamHandler()
        console_handler.setLevel(level)
        console_handler.setFormatter(
            logging.Formatter("%(asctime)s - %(name)s - %(levelname)s - %(message)s")
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

def print_summary(acc_taw, forg_taw):
    """Print summary of results"""
    for name, metric in zip(['TAw Acc','TAw Forg'], [acc_taw, forg_taw]):
        print('*' * 108)
        print(name)
        for i in range(metric.shape[0]):
            print('\t', end='')
            for j in range(metric.shape[1]):
                print('{:5.1f}% '.format(100 * metric[i, j]), end='')
            if np.trace(metric) == 0.0:
                if i > 0:
                    print('\tAvg.:{:5.1f}% '.format(100 * metric[i, :i].mean()), end='')
            else:
                print('\tAvg.:{:5.1f}% '.format(100 * metric[i, :i + 1].mean()), end='')
            print()
    print('*' * 108)

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