import os
import random
import logging
import numpy as np
from PIL import Image

import torch
from torch.utils.data import Dataset

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