import time
import argparse
import numpy as np
from copy import deepcopy

import torch
import torch.nn as nn
import torch.optim as optim
import torch.optim.lr_scheduler as lr_scheduler
from torch.utils.data import DataLoader

from utils import *
from models.base import NetHead
from models.vgg16 import VGG16
from models.spikevgg9 import SpikeVGG9

model_conf = {
    'vgg16': VGG16,
    'svgg9': SpikeVGG9
}

parser = argparse.ArgumentParser(description='Simulate update stage for ECC-SNN')
parser.add_argument('-ee',
                    '--edge_epochs',
                    default=70,
                    type=int,
                    metavar='N',
                    help='number of total epochs to run edge model')
parser.add_argument('-patience', 
                    '--lr-patience', 
                    default=10, 
                    type=int, 
                    required=False,
                    help='Maximum patience to wait before decreasing learning rate')
parser.add_argument('-b',
                    '--batch_size',
                    default=64,
                    type=int,
                    metavar='N',
                    help='mini-batch size')
parser.add_argument('-seed',
                    '--seed',
                    default=2025,
                    type=int,
                    help='seed for initializing training.')
parser.add_argument('-gpu',
                    '--gpu_id',
                    default=6,
                    type=int,
                    help='GPU ID to use')
parser.add_argument('-T',
                    default=4,
                    type=int,
                    metavar='N',
                    help='snn simulation time (default: 2)')
parser.add_argument('-dataset',
                    '--dataset',
                    default='cifar100',
                    type=str,
                    help='cifar10, cifar100')
parser.add_argument('-edge',
                    default='svgg9',
                    type=str,
                    help='edge model name')
parser.add_argument('-base', 
                    '--nc-first-task', 
                    default=None, 
                    type=int, 
                    required=False,
                    help='Number of classes of the first task')
parser.add_argument('-stop', 
                    '--stop-at-task', 
                    default=0, 
                    type=int, 
                    required=False,
                    help='Stop training after specified task')
parser.add_argument('-nt', 
                    '--num-tasks', 
                    default=5, 
                    type=int, 
                    help='Number of tasks')
parser.add_argument('-fix-bn', 
                    '--fix-bn', 
                    action='store_true',
                    help='Fix batch normalization after first task')
parser.add_argument('-distill', 
                    action='store_true', 
                    help='train edge with distillation or directly')


args = parser.parse_args()
print(args)
seed_all(args.seed)

trn_load = torch.load(f'saved/train_loader_{args.dataset}_base{args.nc_first_task}_task{args.num_tasks}.pt')
tst_load = torch.load(f'saved/test_loader_{args.dataset}_base{args.nc_first_task}_task{args.num_tasks}.pt')
taskcla = torch.load(f'saved/taskcla_{args.dataset}_base{args.nc_first_task}_task{args.num_tasks}.pt')


# TODO