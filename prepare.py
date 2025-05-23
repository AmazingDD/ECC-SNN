import time
import math
import argparse
import numpy as np
from tqdm import tqdm
from copy import deepcopy

import torch
import torch.nn as nn
import torch.optim as optim
import torch.optim.lr_scheduler as lr_scheduler
from torch.utils.data import DataLoader

import torchvision
import torchvision.transforms as transforms

from spikingjelly.datasets import split_to_train_test_set

from utils import *
from models.base import NetHead
from models.vgg import VGG16
from models.spikevgg import SpikeVGG9
from models.resnet import *
from models.spikeresnet import *
from models.spikevit import SVIT
from models.vit import *

model_conf = {
    'vgg': VGG16,
    'svgg': SpikeVGG9,
    'resnet50': resnet50,
    'resnet34': resnet34,
    'svit': SVIT,
    'vit': VIT,
    'sresnet34': sew_resnet34,
    'sresnet18': sew_resnet18,
    'sresnet10': sew_resnet10,
    'vit4': VIT4,
}

logger = Logger(
    name="prepare.py", 
    log_file=f"prepare_{get_local_time()}.log", 
    level=logging.INFO).get_logger()

parser = argparse.ArgumentParser(description='Setup stage for ECC-SNN')
parser.add_argument('-j',
                    '--workers',
                    default=4,
                    type=int,
                    metavar='N',
                    help='number of data loading workers (default: 10)')
parser.add_argument('-ce',
                    '--cloud_epochs',
                    default=30, # 200 for training from scratch, 30 for fine tune
                    type=int,
                    metavar='N',
                    help='number of total epochs to run cloud model')
parser.add_argument('-ee',
                    '--edge_epochs',
                    default=200, 
                    type=int,
                    metavar='N',
                    help='number of total epochs to run edge model')
parser.add_argument('-patience', 
                    '--lr-patience', 
                    default=40, 
                    type=int, 
                    required=False,
                    help='Maximum patience to wait before decreasing learning rate')
parser.add_argument('-b',
                    '--batch_size',
                    default=64, # 128
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
                    default='imagenet', # cifar100
                    type=str,
                    help='dataset name')
parser.add_argument('-cloud',
                    default='vit', # vgg
                    type=str,
                    help='cloud model name')
parser.add_argument('-edge',
                    default='svgg', # svgg
                    type=str,
                    help='edge model name')
parser.add_argument('-base', 
                    '--nc-first-task', 
                    default=0, 
                    type=int, 
                    help='Number of classes of the first task')
parser.add_argument('-nt', 
                    '--num-tasks', 
                    default=10, 
                    type=int, 
                    help='Number of tasks')
parser.add_argument('-fix-bn', 
                    '--fix-bn', 
                    action='store_true',
                    help='Fix batch normalization after first task')
parser.add_argument('-pretrain', 
                    action='store_true',
                    help='using pretrained model for imagenet')
parser.add_argument('-distill', 
                    action='store_true', 
                    help='train edge with distillation or directly')
parser.add_argument('-l1',
                    type=float,
                    default=0.5,
                    help='logit distillation intensity')
parser.add_argument('-l2',
                    type=float,
                    default=0.,
                    help='feature distillation intensity')
parser.add_argument('-temperature', 
                    type=float, 
                    default=3.0, 
                    help='Temperature for logit distillation')
args = parser.parse_args()
logger.info(args)
seed_all(args.seed)

# ensure path to save model
ensure_dir(f'saved/{args.dataset}/base{args.nc_first_task}_task{args.num_tasks}')

device = torch.device(f"cuda:{args.gpu_id}" if torch.cuda.is_available() else "cpu")
logger.info(f'Device: {device}')

if args.dataset == 'cifar10':
    transform_train = transforms.Compose([
        transforms.RandomCrop(32, padding=4),
        transforms.RandomHorizontalFlip(),
        transforms.ToTensor(),
        transforms.Normalize((0.4914, 0.4822, 0.4465), (0.2023, 0.1994, 0.2010)),
    ])
    transform_test = transforms.Compose([
        transforms.ToTensor(),
        transforms.Normalize((0.4914, 0.4822, 0.4465), (0.2023, 0.1994, 0.2010)),
    ])
    train_set = torchvision.datasets.CIFAR10(root='.', train=True, download=True, transform=transform_train)
    test_set = torchvision.datasets.CIFAR10(root= '.', train=False, download=False, transform=transform_test)

    num_classes = 10
    C, H, W = 3, 32, 32

elif args.dataset == 'cifar100':
    transform_train = transforms.Compose([
        transforms.RandomCrop(32, padding=4),
        transforms.RandomHorizontalFlip(),
        transforms.ToTensor(),
        transforms.Normalize((0.5071, 0.4867, 0.4408), (0.2675, 0.2565, 0.2761)),
    ])
    transform_test = transforms.Compose([
        transforms.ToTensor(),
        transforms.Normalize((0.5071, 0.4867, 0.4408), (0.2675, 0.2565, 0.2761)),
    ])
    train_set = torchvision.datasets.CIFAR100(root='.', train=True, download=True, transform=transform_train)
    test_set = torchvision.datasets.CIFAR100(root= '.', train=False, download=False, transform=transform_test)

    num_classes = 100
    C, H, W = 3, 32, 32

elif args.dataset == 'caltech':
    transform_train = transforms.Compose([
        # transforms.Resize((224, 224), interpolation=transforms.InterpolationMode.BICUBIC),
        transforms.RandomHorizontalFlip(), 
        transforms.ToTensor(),
        transforms.Normalize(mean=(0.485, 0.456, 0.406), std=(0.229, 0.224, 0.225))
    ])

    transform_test = transforms.Compose([
        # transforms.Resize((224, 224), interpolation=transforms.InterpolationMode.BICUBIC),
        transforms.RandomHorizontalFlip(), 
        transforms.ToTensor(),
        transforms.Normalize(mean=(0.485, 0.456, 0.406), std=(0.229, 0.224, 0.225))
    ])

    dset = CaltechDataset(root='.', transform=transform_train)
    
    num_classes = 101
    C, H, W = 3, 224, 224

    train_set, test_set = split_to_train_test_set(0.8, dset, num_classes)

elif args.dataset == 'imagenet':
    transform_train = transforms.Compose([
        transforms.Resize(224, interpolation=transforms.InterpolationMode.BICUBIC),
        transforms.ToTensor(),
        transforms.Normalize(mean=(0.485, 0.456, 0.406), std=(0.229, 0.224, 0.225))
    ])

    transform_test = transforms.Compose([
        transforms.Resize(224, interpolation=transforms.InterpolationMode.BICUBIC),
        transforms.ToTensor(),
        transforms.Normalize(mean=(0.485, 0.456, 0.406), std=(0.229, 0.224, 0.225))
    ])

    num_classes = 200
    C, H, W = 3, 224, 224

    train_set = TinyImageNetDataset('./tiny-imagenet-200', train=True, transform=transform_train)
    test_set = TinyImageNetDataset('./tiny-imagenet-200', train=False, transform=transform_test)
else:
    raise NotImplementedError(f'Invalid dataset name: {args.dataset}')

############################################################################################
############### Preparing Cloud ANN model with all dataset to make it Oracle ###############
############################################################################################

if os.path.exists(f'saved/{args.dataset}/base{args.nc_first_task}_task{args.num_tasks}/best_cloud_{args.cloud}.pt'):
    logger.info(f'Already have trained cloud model {args.cloud} for {args.dataset} Base{args.nc_first_task}-Inc{args.num_tasks}')
else:
    logger.info('Training cloud ANN')

    trainloader = DataLoader(train_set, batch_size=args.batch_size, shuffle=True, num_workers=args.workers)
    testloader = DataLoader(test_set, batch_size=args.batch_size, shuffle=False, num_workers=args.workers)

    if args.pretrain: # args.cloud in ('vit', 'vgg', 'resnet34')
        model = model_conf[args.cloud](num_classes, C, H, W, args.T, args.pretrain)
    else:
        model = model_conf[args.cloud](num_classes, C, H, W, args.T)
    print(model)

    model.to(device)

    n_parameters = sum(p.numel() for p in model.parameters() if hasattr(p, 'requires_grad'))
    logger.info(f"number of params for cloud model: {n_parameters}")

    criterion = nn.CrossEntropyLoss()

    if args.pretrain:
        # fine-tune
        optimizer = optim.AdamW(model.classifier.parameters(), lr=1e-3, weight_decay=0.05)
        scheduler = lr_scheduler.CosineAnnealingLR(optimizer, T_max=math.ceil(len(trainloader) / 2) * args.cloud_epochs)
    elif args.cloud in ('vit', 'vit4') and not args.pretrain:
        # vit from scratch is special
        optimizer = optim.AdamW(model.classifier.parameters(), lr=5e-4, weight_decay=0.05)
        scheduler = lr_scheduler.CosineAnnealingLR(optimizer, T_max=args.cloud_epochs)
    else: 
        # training from scratch
        optimizer = optim.SGD(model.parameters(), lr=0.1, weight_decay=5e-4, momentum=0.9)
        scheduler = lr_scheduler.MultiStepLR(optimizer, milestones=[60, 120, 160], gamma=0.2)

    best_acc = 0.
    for epoch in range(args.cloud_epochs):
        model.train()
        running_loss = 0.
        for images, labels in tqdm(trainloader, unit='batch'):
            optimizer.zero_grad()

            images, labels = images.to(device), labels.to(device)

            if args.pretrain: # all pretrain model input size is 3*224*224
                images = nn.functional.interpolate(images, size=(224, 224), mode='bilinear', align_corners=False)

            logits, _ = model(images)
            loss = criterion(logits, labels)
            loss.backward()
            optimizer.step()

            running_loss += loss.item()

        scheduler.step()
        logger.info(f'Epoch [{epoch + 1}/ {args.cloud_epochs}], Loss: {running_loss / len(trainloader)}')

        model.eval()
        correct, total = 0, 0
        with torch.no_grad():
            for images, labels in testloader:
                images, labels = images.to(device), labels.to(device)

                if args.pretrain: # all pretrain model input size is 3*224*224
                    images = nn.functional.interpolate(images, size=(224, 224), mode='bilinear', align_corners=False)

                logits, _ = model(images)

                _, predicted = torch.max(logits.data, 1)
                total += labels.size(0)
                correct += (predicted == labels).sum().item()

        acc = 100 * correct / total
        logger.info(f'Test Accuracy on {args.dataset} for cloud {args.cloud}: {acc:.2f}%')

        if acc > best_acc:
            best_acc = acc
            torch.save(deepcopy(model.state_dict()), f'saved/{args.dataset}/base{args.nc_first_task}_task{args.num_tasks}/best_cloud_{args.cloud}.pt')

    logger.info(f'Finished preparing cloud model with best test accuracy {best_acc}%...')

#########################################################################################
####################### prepare incremental scenario for edge SNN #######################
#########################################################################################

if 'cifar10' in args.dataset: # cifar10, cifar100 need the same process method
    trn_data = {'x': train_set.data, 'y': train_set.targets}
    tst_data = {'x': test_set.data, 'y': test_set.targets}
elif 'imagenet' in args.dataset:
    trn_data = {'x': train_set.data.permute(0, 2, 3, 1).numpy(), 'y': train_set.targets.tolist()} 
    tst_data = {'x': test_set.data.permute(0, 2, 3, 1).numpy(), 'y': test_set.targets.tolist()}
elif 'caltech' in args.dataset:
    train_idx = train_set.indices
    test_idx = test_set.indices
    trn_data = {'x': train_set.dataset.data[train_idx], 'y': train_set.dataset.targets[train_idx].tolist()} 
    tst_data = {'x': test_set.dataset.data[test_idx], 'y': test_set.dataset.targets[test_idx].tolist()}
else:
    raise ValueError(f'the selected dataset {args.dataset} is to be added')

data = {}
taskcla = [] # record each task contains how many labels

num_tasks = args.num_tasks
nc_first_task = args.nc_first_task

if args.dataset == 'caltech':
    # sort by frequency
    class_order = [  5,   3,   0,   1,  94,   2,  12,  19,  55,  23,  47,  46,  13,
                    16,  50,  63,  54,  86,  92,  15,  39,  90,  81,  75,  57,  51,
                    58,  65,  35,  93,  26,  27,  25,  34,  31,  40,  41,  60,  33,
                    36,  38,  53,  84,  88,  79,  22,  56, 100,  21,  76,  87,  96,
                    30,  74,  82,  98,   4,  66,   9,  49,  37,  72,  32,  29,  45,
                    17,  28,  77,  91,   8,  20,  24,  69,  10,  42,  71,  85,  14,
                    18,  61,   6,   7,  48,  59,  62,  78,  68,  80,  99,  70,  95,
                    67,  83,  89,  43,  44,  73,  97,  11,  64,  52]
else:
    class_order = list(range(num_classes))
    np.random.shuffle(class_order)

if nc_first_task == 0: 
    # no number of classes is assigned for base task 0, labels are evenly divided by num_tasks
    cpertask = np.array([num_classes // num_tasks] * num_tasks)
    for i in range(num_classes % num_tasks):
        cpertask[i] += 1
else: 
    # remaining labels are evenly divided by (num_tasks - 1)
    assert nc_first_task < num_classes, "first task wants more classes than exist"
    remaining_classes = num_classes - nc_first_task
    assert remaining_classes >= (num_tasks - 1), "at least one class is needed per task"
    cpertask = np.array([nc_first_task] + [remaining_classes // (num_tasks - 1)] * (num_tasks - 1))
    for i in range(remaining_classes % (num_tasks - 1)):
        cpertask[i + 1] += 1
assert num_classes == cpertask.sum(), "something went wrong, the split does not match num classes"

cpertask_cumsum = np.cumsum(cpertask) # e.g. [60, 10, 10, 10, 10] -> [60, 70, 80, 90, 100]
init_class = np.concatenate(([0], cpertask_cumsum[:-1])) # e.g. [0, 60, 70, 80, 90]

# initialize data structure
for tt in range(num_tasks):
    data[tt] = {}
    data[tt]['name'] = 'task-' + str(tt)
    data[tt]['trn'] = {'x': [], 'y': []}
    data[tt]['tst'] = {'x': [], 'y': []}

# filter those samples with labels not in class_order
filtering = np.isin(trn_data['y'], class_order)
if filtering.sum() != len(trn_data['y']):
    trn_data['x'] = trn_data['x'][filtering]
    trn_data['y'] = np.array(trn_data['y'])[filtering]
filtering = np.isin(tst_data['y'], class_order)
if filtering.sum() != len(tst_data['y']):
    tst_data['x'] = tst_data['x'][filtering]
    tst_data['y'] = tst_data['y'][filtering]

# add data to each task with reindexed label
for this_image, this_label in zip(trn_data['x'], trn_data['y']):
    # the new label is the index for origin label in the shuffled class_order
    this_label = class_order.index(this_label)
    this_task = (this_label >= cpertask_cumsum).sum()
    data[this_task]['trn']['x'].append(this_image)
    # e.g. after reindex, its label is 61, it belongs to task 1, but for task 1, its label is just 1
    data[this_task]['trn']['y'].append(this_label - init_class[this_task]) 
for this_image, this_label in zip(tst_data['x'], tst_data['y']):
    this_label = class_order.index(this_label)
    this_task = (this_label >= cpertask_cumsum).sum()
    data[this_task]['tst']['x'].append(this_image)
    data[this_task]['tst']['y'].append(this_label - init_class[this_task])

# check classes
for tt in range(num_tasks):
    # ncla is the number of class for current task 
    data[tt]['ncla'] = len(np.unique(data[tt]['trn']['y']))
    assert data[tt]['ncla'] == cpertask[tt], "something went wrong splitting classes" 

# convert to numpy arrays
for tt in data.keys():
    for split in ['trn', 'tst']:
        data[tt][split]['x'] = np.asarray(data[tt][split]['x'])

# counting classes number information
n = 0
for tt in data.keys():
    taskcla.append((tt, data[tt]['ncla']))
    n += data[tt]['ncla']
data['ncla'] = n

trn_dset, tst_dset = [], []
offset = 0
for task in range(num_tasks):
    data[task]['trn']['y'] = [label + offset for label in data[task]['trn']['y']]
    data[task]['tst']['y'] = [label + offset for label in data[task]['tst']['y']]
    trn_dset.append(BaseDataset(data[task]['trn'], transform_train, class_order))
    tst_dset.append(BaseDataset(data[task]['tst'], transform_test, class_order))

    offset += taskcla[task][1] # [task][1] is data[tt]['ncla'], which is the number of labels in task tt

# get dataloader for each task
trn_load, tst_load = [], []
for tt in range(num_tasks):
    trn_load.append(DataLoader(trn_dset[tt], 
                                batch_size=args.batch_size, 
                                shuffle=True, 
                                num_workers=args.workers, 
                                pin_memory=True))
    tst_load.append(DataLoader(tst_dset[tt], 
                                batch_size=args.batch_size, 
                                shuffle=False, 
                                num_workers=args.workers, 
                                pin_memory=True))
if args.dataset in ('imagenet'):
    # our GPU memory is limited, for SNN model with T=4, it cannot handle 224 input size
    init_model = model_conf[args.edge](num_classes, C, 64, 64, T=args.T)
else:
    init_model = model_conf[args.edge](num_classes, C, H, W, T=args.T)

# base edge SNN
seed_all(args.seed)
net = NetHead(init_model)
seed_all(args.seed)
net_old = None

n_parameters = sum(p.numel() for p in net.parameters() if hasattr(p, 'requires_grad'))
logger.info(f"number of params for base edge model: {n_parameters}")

for t, (_, ncla) in enumerate(taskcla): # task 0->n, but only task 0 in prepare stage
    if t > 0: 
        break # we only consider the task 0 for the base model

    print('*' * 108)
    logger.info(f'Task {t:2d}')
    print('*' * 108)

    net.add_head(taskcla[t][1]) 
    net.to(device)

    if not args.distill:
        logger.info('Directly training edge SNN')

        if 'sresnet' in args.edge and args.dataset in ('imagenet'):
            # sew-resnet default optimizer
            optimizer = optim.SGD(net.parameters(), lr=0.1)
        else:
            # common SNN optimizer
            optimizer = optim.Adam(net.parameters(), lr=1e-3, weight_decay=5e-4)
        scheduler = lr_scheduler.CosineAnnealingLR(optimizer, T_max=args.edge_epochs)
        criterion = nn.CrossEntropyLoss()

        best_acc = -np.inf
        patience = args.lr_patience
        best_model = net.get_copy()
        for e in range(args.edge_epochs):
            clock0 = time.time()
            net.train()
            if args.fix_bn and t > 0:
                net.freeze_bn()
            for images, targets in tqdm(trn_load[t], unit='batch'):
                if args.dataset in ('imagenet'):
                    images = nn.functional.interpolate(images, size=(64, 64), mode='bilinear', align_corners=False)
                outputs, _ = net(images.to(device))
                loss = criterion(outputs[t], targets.to(device) - net.task_offset[t])
                
                optimizer.zero_grad()
                loss.backward()
                nn.utils.clip_grad_norm_(net.parameters(), 10000)
                optimizer.step()
            scheduler.step()
            clock1 = time.time()

            line = f'Epoch {e + 1:3d}, train time={clock1 - clock0:5.1f}s, '

            clock3 = time.time()
            with torch.no_grad():
                total_loss, total_acc, total = 0, 0, 0
                net.eval()
                for images, targets in tst_load[t]:
                    if args.dataset in ('imagenet'):
                        images = nn.functional.interpolate(images, size=(64, 64), mode='bilinear', align_corners=False)
                    outputs, _ = net(images.to(device))
                    loss = criterion(outputs[t], targets.to(device) - net.task_offset[t])
                    # calculate batch accuracy 
                    pred = torch.zeros_like(targets.to(device))
                    for m in range(len(pred)):
                        this_task = (net.task_cls.cumsum(0) <= targets[m]).sum()
                        pred[m] = outputs[this_task][m].argmax() + net.task_offset[this_task]
                    acc = (pred == targets.to(device)).float()

                    total_loss += loss.item() * len(targets)
                    total += len(targets)
                    total_acc += acc.sum().item()
                test_loss, test_acc = total_loss / total, total_acc / total
            clock4 = time.time()
            line += f'test time={clock4 - clock3:5.2f}s, loss={test_loss:.3f}, test acc={100 * test_acc:5.2f}%'

            if test_acc >= best_acc:
                best_acc = test_acc
                best_model = net.get_copy()
                patience = args.lr_patience
                line += ' *'
            else:
                patience -= 1
                if patience <= 0:
                    net.set_state_dict(best_model)
                    logger.info(line)
                    break
            logger.info(line)

        net.set_state_dict(best_model)

        # save base edge model
        torch.save(net.get_copy(), f'saved/{args.dataset}/base{args.nc_first_task}_task{args.num_tasks}/best_edge_base_{args.edge}.pt')

    else:
        total_cls_in_t = net.task_cls.sum().item()
        cloud_label_index = class_order[:total_cls_in_t]

        logger.info('Training edge SNN assisted by cloud ANN distillation')
        # init cloud model with pre-trained weight, then remove head and finetune for new base task
        if args.pretrain:
            c_net = model_conf[args.cloud](num_classes, C, H, W, args.T, args.pretrain)
        else:
            c_net = model_conf[args.cloud](num_classes, C, H, W, args.T)
        c_net.load_state_dict(
            torch.load(f'saved/{args.dataset}/base{args.nc_first_task}_task{args.num_tasks}/best_cloud_{args.cloud}.pt', map_location='cpu'))
        c_net.to(device)
        c_net.eval()
        logger.info('cloud model loaded successfully...')

        if 'sresnet' in args.edge and args.dataset in ('imagenet'):
            # sew-resnet default optimizer
            optimizer = optim.SGD(net.parameters(), lr=0.1)
        else:
            # SNN default optimizer
            optimizer = optim.Adam(net.parameters(), lr=1e-3, weight_decay=5e-4)
        scheduler = lr_scheduler.CosineAnnealingLR(optimizer, T_max=args.edge_epochs)
        criterion = nn.CrossEntropyLoss()
        
        best_acc = -np.inf
        patience = args.lr_patience
        best_model = net.get_copy()
        for e in range(args.edge_epochs):
            clock0 = time.time()
            net.train()
            if args.fix_bn and t > 0:
                net.freeze_bn()
            for images, targets in tqdm(trn_load[t], unit='batch'):
                images = images.to(device) 
                targets = targets.to(device)

                # cloud infer
                if args.pretrain:
                    c_logit, c_feature = c_net(nn.functional.interpolate(images, size=(224, 224), mode='bilinear', align_corners=False))
                else:
                    c_logit, c_feature = c_net(images)
                # select those labels occured in current task, and convert them to new index
                c_logit = c_logit[:, cloud_label_index] 
                
                # edge infer
                if args.dataset in ('imagenet'):
                    images = nn.functional.interpolate(images, size=(64, 64), mode='bilinear', align_corners=False)
                e_outputs, e_feature = net(images)

                # ce loss
                loss = criterion(e_outputs[t], targets.to(device) - net.task_offset[t])

                # logit loss
                soft_targets = nn.functional.softmax(c_logit / args.temperature, dim=1)
                soft_logits = nn.functional.log_softmax(e_outputs[t] / args.temperature, dim=1)
                loss_logit = nn.functional.kl_div(soft_logits, soft_targets, reduction='batchmean') * (args.temperature ** 2)
                loss += args.l1 * loss_logit

                # feature loss if overlapping case
                if (args.cloud == 'vgg' and args.edge == 'svgg'):
                    loss_align = nn.functional.mse_loss(e_feature, c_feature)
                    loss += args.l2 * loss_align

                optimizer.zero_grad()
                loss.backward()
                nn.utils.clip_grad_norm_(net.parameters(), 10000)
                optimizer.step()

            scheduler.step()
            clock1 = time.time()
            line = f'Epoch {e + 1:3d}, train time={clock1 - clock0:5.1f}s, '

            clock3 = time.time()
            with torch.no_grad():
                total_loss, total_acc, total = 0, 0, 0
                total_c_acc = 0
                net.eval()
                for images, targets in tst_load[t]:
                    # cloud infer
                    if args.pretrain:
                        c_logit, c_feature = c_net(nn.functional.interpolate(images.to(device), size=(224, 224), mode='bilinear', align_corners=False))
                    else:
                        c_logit, _ = c_net(images.to(device))
                    c_logit = c_logit[:, cloud_label_index]

                    # edge infer
                    if args.dataset in ('imagenet'):
                        images = nn.functional.interpolate(images, size=(64, 64), mode='bilinear', align_corners=False)
                    outputs, _ = net(images.to(device))
                    loss = criterion(outputs[t], targets.to(device) - net.task_offset[t])

                    # calculate batch accuracy 
                    pred = torch.zeros_like(targets.to(device))
                    for m in range(len(pred)):
                        this_task = (net.task_cls.cumsum(0) <= targets[m]).sum()
                        pred[m] = outputs[this_task][m].argmax() + net.task_offset[this_task]
                    acc = (pred == targets.to(device)).float()

                    _, c_pred = torch.max(c_logit, 1)
                    c_acc = (c_pred == targets.to(device)).float()

                    total_loss += loss.item() * len(targets)
                    total += len(targets)
                    total_acc += acc.sum().item()

                    total_c_acc += c_acc.sum().item()

                test_loss, test_acc = total_loss / total, total_acc / total
                test_c_acc = total_c_acc / total

            clock4 = time.time()
            line += f'test time={clock4 - clock3:5.2f}s, loss={test_loss:.3f}, test acc={100 * test_acc:5.2f}%, test cloud acc={100 * test_c_acc:5.2f}%'

            if test_acc >= best_acc:
                best_acc = test_acc
                best_model = net.get_copy()
                patience = args.lr_patience
                line +=' *'
            else:
                patience -= 1
                if patience <= 0:
                    net.set_state_dict(best_model)
                    logger.info(line)
                    break
            logger.info(line)
        net.set_state_dict(best_model)

        # save base edge model
        torch.save(net.get_copy(), f'saved/{args.dataset}/base{args.nc_first_task}_task{args.num_tasks}/best_edge_base_{args.edge}.pt')

logger.info(f'Finished preparing edge SNN model with best test accuracy {100 * best_acc:5.2f}%')

torch.save(trn_load, f'saved/{args.dataset}/base{args.nc_first_task}_task{args.num_tasks}/train_loader.pt')
torch.save(tst_load, f'saved/{args.dataset}/base{args.nc_first_task}_task{args.num_tasks}/test_loader.pt')
torch.save(taskcla, f'saved/{args.dataset}/base{args.nc_first_task}_task{args.num_tasks}/taskcla.pt')
torch.save(class_order, f'saved/{args.dataset}/base{args.nc_first_task}_task{args.num_tasks}/classorder.pt')
