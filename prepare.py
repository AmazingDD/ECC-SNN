import time
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

from utils import *
from models.base import NetHead
from models.vgg16 import VGG16
from models.spikevgg9 import SpikeVGG9

model_conf = {
    'vgg16': VGG16,
    'svgg9': SpikeVGG9
}

parser = argparse.ArgumentParser(description='Setup stage for ECC-SNN')
parser.add_argument('-j',
                    '--workers',
                    default=4,
                    type=int,
                    metavar='N',
                    help='number of data loading workers (default: 10)')
parser.add_argument('-ce',
                    '--cloud_epochs',
                    default=200,
                    type=int,
                    metavar='N',
                    help='number of total epochs to run cloud model')
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
parser.add_argument('-cloud',
                    default='vgg16',
                    type=str,
                    help='cloud model name')
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
parser.add_argument('-l1',
                    default=0.2,
                    type=int,
                    metavar='N',
                    help='logit distillation intensity')
parser.add_argument('-l2',
                    default=0.2,
                    type=int,
                    metavar='N',
                    help='feature distillation intensity')
parser.add_argument('-temperature', 
                    type=float, 
                    default=3.0, 
                    help='Temperature for logit distillation')
args = parser.parse_args()
print(args)
seed_all(args.seed)

# ensure path to save model
ensure_dir('saved/')

device = torch.device(f"cuda:{args.gpu_id}" if torch.cuda.is_available() else "cpu")

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

elif args.dataset == 'tiny-imagenet':
    # TODO
    num_classes = 200
    C, H, W = 3, 128, 128
else:
    raise NotImplementedError(f'Invalid dataset name: {args.dataset}')

############################################################################################
############### Preparing Cloud ANN model with all dataset to make it Oracle ###############
############################################################################################
print('Training cloud ANN')

trainloader = DataLoader(train_set, batch_size=args.batch_size, shuffle=True, num_workers=args.workers)
testloader = DataLoader(test_set, batch_size=args.batch_size, shuffle=False, num_workers=args.workers)

model = model_conf[args.cloud](num_classes, C, H, W)
model.to(device)

criterion = nn.CrossEntropyLoss()
optimizer = optim.SGD(model.parameters(), lr=0.1, weight_decay=5e-4, momentum=0.9)
scheduler = lr_scheduler.MultiStepLR(optimizer, milestones=[60, 120, 160], gamma=0.2)

best_acc = 0.
for epoch in range(args.cloud_epochs):
    model.train()
    running_loss = 0.
    for images, labels in tqdm(trainloader, unit='batch'):
        optimizer.zero_grad()

        images, labels = images.to(device), labels.to(device)
        logits, _ = model(images)
        loss = criterion(logits, labels)
        loss.backward()
        optimizer.step()

        running_loss += loss.item()

    scheduler.step()
    print(f'Epoch [{epoch + 1}/ {args.cloud_epochs}], Loss: {running_loss / len(trainloader)}')

    model.eval()
    correct, total = 0, 0
    with torch.no_grad():
        for images, labels in testloader:
            images, labels = images.to(device), labels.to(device)
            logits, _ = model(images)

            _, predicted = torch.max(logits.data, 1)
            total += labels.size(0)
            correct += (predicted == labels).sum().item()

    acc = 100 * correct / total
    print(f'Test Accuracy on {args.dataset} for cloud {args.cloud}: {acc:.2f}%')

    if acc > best_acc:
        best_acc = acc
        torch.save(deepcopy(model.state_dict()), f'saved/best_cloud_{args.cloud}_{args.dataset}.pt')

print(f'Finished preparing cloud model with best test accuracy {best_acc}%...')

#########################################################################################
####################### prepare incremental scenario for edge SNN #######################
#########################################################################################
if 'mnist' in args.dataset:
    trn_data = {'x': train_set.data.numpy(), 'y': train_set.targets.tolist()}
    tst_data = {'x': test_set.data.numpy(), 'y': test_set.targets.tolist()}
elif 'cifar' in args.dataset:
    trn_data = {'x': train_set.data, 'y': train_set.targets}
    tst_data = {'x': test_set.data, 'y': test_set.targets}
elif 'imagenet' in args.dataset:
    pass
else:
    raise ValueError(f'the selected dataset {args.dataset} is to be added')

data = {}
taskcla = [] # record each task contains how many labels

num_tasks = args.num_tasks
nc_first_task = args.nc_first_task

class_order = list(range(num_classes))
np.random.shuffle(class_order)

if nc_first_task is None:
    cpertask = np.array([num_classes // num_tasks] * num_tasks)
    for i in range(num_classes % num_tasks):
        cpertask[i] += 1
else:
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
    
# max_task = len(taskcla) if args.stop_at_task == 0 else args.stop_at_task

init_model = model_conf[args.edge](num_classes, C, H, W)
init_model.T = args.T

# base edge SNN
seed_all(args.seed)
net = NetHead(init_model)
seed_all(args.seed)
net_old = None

for t, (_, ncla) in enumerate(taskcla): # task 0->n, but only task 0 in prepare stage
    if t > 0: 
        break # we only consider the task 0 for the base model

    print('*' * 108)
    print(f'Task {t:2d}')
    print('*' * 108)

    net.add_head(taskcla[t][1]) 
    net.to(device)

    if not args.distill:
        print('Directly training edge SNN')

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
            for images, targets in trn_load[t]:
                outputs, _ = net(images.to(device))
                loss = criterion(outputs[t], targets.to(device) - net.task_offset[t])
                
                optimizer.zero_grad()
                loss.backward()
                nn.utils.clip_grad_norm_(net.parameters(), 10000)
                optimizer.step()
            scheduler.step()
            clock1 = time.time()
            print(f'| Epoch {e + 1:3d}, train time={clock1 - clock0:5.1f}s |', end='')

            clock3 = time.time()
            with torch.no_grad():
                total_loss, total_acc, total = 0, 0, 0
                net.eval()
                for images, targets in tst_load[t]:
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
            print(f' test time={clock4 - clock3:5.2f}s loss={test_loss:.3f}, test acc={100 * test_acc:5.2f}% |', end='')

            if test_acc >= best_acc:
                best_acc = test_acc
                best_model = net.get_copy()
                patience = args.lr_patience
                print(' *', end='')
            else:
                patience -= 1
                if patience <= 0:
                    net.set_state_dict(best_model)
                    break
            print()

        net.set_state_dict(best_model)

        # save base edge model
        torch.save(net.get_copy(), f'saved/best_edge_base_{args.edge}_{args.dataset}.pt')

    else:
        print('Training edge SNN assisted by cloud ANN distillation')
        # init cloud model with pre-trained weight, then remove head and finetune for new base task
        init_model = model_conf[args.cloud](num_classes, C, H, W)
        init_model.load_state_dict(
            torch.load(f'saved/best_cloud_{args.cloud}_{args.dataset}.pt', map_location='cpu'))
        seed_all(args.seed)
        c_net = NetHead(init_model)
        seed_all(args.seed)
        c_net.add_head(taskcla[t][1]) 
        c_net.to(device)
        # this only freeze the cloud model weight, but the classifier_head can update
        c_net.freeze_backbone() 
        print('cloud model loaded successfully...')

        optimizer = optim.Adam(list(net.parameters()) + list(c_net.heads.parameters()), lr=1e-3, weight_decay=5e-4)
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
            for images, targets in trn_load[t]:
                images = images.to(device)
                targets = targets.to(device)
                c_outputs, c_feature = c_net(images)
                e_outputs, e_feature = net(images)

                # ce loss
                loss = criterion(e_outputs[t], targets.to(device) - net.task_offset[t])

                # logit loss
                soft_targets = nn.functional.softmax(c_outputs[t] / args.temperature, dim=1)
                soft_logits = nn.functional.log_softmax(e_outputs[t] / args.temperature, dim=1)
                loss_logit = nn.functional.kl_div(soft_logits, soft_targets, reduction='batchmean') * (args.temperature ** 2)
                loss += args.l1 * loss_logit

                # feature loss if overlapping case
                if (args.cloud == 'vgg16' and args.edge == 'svgg9') or (args.cloud == 'vit4' and args.edge == 'svit1'):
                    loss_align = nn.functional.mse_loss(e_feature, c_feature)
                    loss += args.l2 * loss_align

                optimizer.zero_grad()
                loss.backward()
                nn.utils.clip_grad_norm_(net.parameters(), 10000)
                optimizer.step()
            scheduler.step()
            clock1 = time.time()
            print(f'| Epoch {e + 1:3d}, train time={clock1 - clock0:5.1f}s |', end='')

            clock3 = time.time()
            with torch.no_grad():
                total_loss, total_acc, total = 0, 0, 0
                net.eval()
                for images, targets in tst_load[t]:
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
            print(f' test time={clock4 - clock3:5.2f}s loss={test_loss:.3f}, test acc={100 * test_acc:5.2f}% |', end='')

            if test_acc >= best_acc:
                best_acc = test_acc
                best_model = net.get_copy()
                patience = args.lr_patience
                print(' *', end='')
            else:
                patience -= 1
                if patience <= 0:
                    net.set_state_dict(best_model)
                    break
            print()
        net.set_state_dict(best_model)

        # save base edge model
        torch.save(net.get_copy(), f'saved/best_edge_base_{args.edge}_{args.dataset}.pt')

    # post process for lwf, preparing for the next task
    # net_old = deepcopy(net)
    # net_old.eval()
    # net_old.freeze_all()
print(f'Finished preparing edge SNN model with best test accuracy {100 * best_acc:5.2f}%...')

torch.save(trn_load, f'saved/train_loader_{args.dataset}_base{args.nc_first_task}_task{args.num_tasks}.pt')
torch.save(tst_load, f'saved/test_loader_{args.dataset}_base{args.nc_first_task}_task{args.num_tasks}.pt')
torch.save(taskcla, f'saved/taskcla_{args.dataset}_base{args.nc_first_task}_task{args.num_tasks}.pt')
torch.save(class_order, f'saved/classorder_{args.dataset}_base{args.nc_first_task}_task{args.num_tasks}.pt')
