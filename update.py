import time
import argparse
import numpy as np
from copy import deepcopy

import torch
import torch.nn as nn
import torch.optim as optim
import torch.optim.lr_scheduler as lr_scheduler

from utils import *
from models.base import NetHead
from models.vgg16 import VGG16
from models.spikevgg9 import SpikeVGG9

model_conf = {
    'vgg16': VGG16,
    'svgg9': SpikeVGG9
}

def self_logit_distill(outputs, targets, temperature):
        """self-distillation with temperature scaling"""
        new_soft_logits = nn.functional.log_softmax(outputs / temperature, dim=1)
        old_soft_logits = nn.functional.softmax(targets / temperature, dim=1)
        l_old = nn.functional.kl_div(new_soft_logits, old_soft_logits, reduction='batchmean') * (t ** 2)

        return l_old

tstart = time.time()
logger = Logger(name="update.py", log_file="update.log", level=logging.INFO).get_logger()

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
parser.add_argument('-lamb', 
                    default=0.5, 
                    type=float, 
                    help='regularization for L_old')
parser.add_argument('-temperature', 
                    default=2.0, 
                    type=float, 
                    help='distillation temperature')
args = parser.parse_args()
logger.info(args)
seed_all(args.seed)
device = torch.device(f"cuda:{args.gpu_id}" if torch.cuda.is_available() else "cpu")

trn_load = torch.load(f'saved/train_loader_{args.dataset}_base{args.nc_first_task}_task{args.num_tasks}.pt')
tst_load = torch.load(f'saved/test_loader_{args.dataset}_base{args.nc_first_task}_task{args.num_tasks}.pt')
taskcla = torch.load(f'saved/taskcla_{args.dataset}_base{args.nc_first_task}_task{args.num_tasks}.pt')

max_task = len(taskcla) if args.stop_at_task == 0 else args.stop_at_task

# two matrix for final results
acc_taw = np.zeros((max_task, max_task))
forg_taw = np.zeros((max_task, max_task))

if args.dataset == 'cifar100':
    num_classes = 100
    C, H, W = 3, 32, 32
elif args.dataset == 'cifar10':
    num_classes = 10
    C, H, W = 3, 32, 32
elif 'imagenet' in args.dataset:
    pass
else:
    raise NotImplementedError(f'Invalid dataset name: {args.dataset}')

# load base edge SNN
init_model = model_conf[args.edge](num_classes, C, H, W)
init_model.T = args.T
seed_all(args.seed)
net = NetHead(init_model)
seed_all(args.seed)

# add base task head
net.add_head(taskcla[0][1]) 
net.set_state_dict(torch.load(f'saved/best_edge_base_{args.edge}_{args.dataset}.pt', map_location='cpu'))
net.to(device)

# post process for lwf, preparing for the next task, start from task 1
net_old = deepcopy(net)
net_old.eval()
net_old.freeze_all()

for t, (_, ncla) in enumerate(taskcla): # task 0->n
    if t >= max_task:
        continue

    print('*' * 108)
    logger.info(f'Task {t:2d}')
    print('*' * 108)

    if t > 0:
        net.add_head(taskcla[t][1]) 
        net.to(device)

        # if there are no exemplars, previous heads are not modified
        if len(net.heads) > 1:
            params = list(net.model.parameters()) + list(net.heads[-1].parameters())
        else:
            params = net.parameters()
        optimizer = optim.Adam(params, lr=1e-3, weight_decay=0.)
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
                outputs_old = None
                if t > 0:
                    outputs_old, _ = net_old(images.to(device))

                outputs, _ = net(images.to(device))

                # L_new
                loss = criterion(outputs[t], targets.to(device) - net.task_offset[t])
                if t > 0:
                    # \alpha * L_old
                    loss += args.lamb * self_logit_distill(
                        torch.cat(outputs[:t], dim=1), 
                        torch.cat(outputs_old[:t], dim=1), 
                        args.temperature)
                    
                optimizer.zero_grad()
                loss.backward()
                nn.utils.clip_grad_norm_(net.parameters(), 10000)
                optimizer.step()

            scheduler.step()
            clock1 = time.time()

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

            rec_str = f'Epoch {e + 1:3d}, train time={clock1 - clock0:5.1f}s, test time={clock4 - clock3:5.2f}s, loss={test_loss:.3f}, test acc={100 * test_acc:5.2f}%'

            if test_acc >= best_acc:
                best_acc = test_acc
                best_model = net.get_copy()
                patience = args.lr_patience
                rec_str += ' *'
            else:
                patience -= 1
                if patience <= 0:
                    net.set_state_dict(best_model)
                    logger.info(rec_str)
                    break
            logger.info(rec_str)
        net.set_state_dict(best_model)

        net_old = deepcopy(net)
        net_old.eval()
        net_old.freeze_all()

    else:
        logger.info('already finish task 0 at prepare.py...')
    
    print('-' * 108)

    # CIL Test 
    res_out=''
    for u in range(t + 1):
        with torch.no_grad():
            total_acc_taw, total_taw = 0, 0
            net.eval()
            for images, targets in tst_load[u]:
                outputs, _ = net(images.to(device))
                pred = torch.zeros_like(targets.to(device))
                for m in range(len(pred)):
                    this_task = (net.task_cls.cumsum(0) <= targets[m]).sum()
                    pred[m] = outputs[this_task][m].argmax() + net.task_offset[this_task]
                acc = (pred == targets.to(device)).float()
                total_taw += len(targets)
                total_acc_taw += acc.sum().item()
            test_acc_taw = total_acc_taw / total_taw
        
        acc_taw[t, u] = test_acc_taw
        if u < t:
            forg_taw[t, u] = acc_taw[:t, u].max(0) - acc_taw[t, u]
        res_tmp = f'>>> Test on task {u:2d} | TAw acc={100 * acc_taw[t, u]:5.1f}%, forg={100 * forg_taw[t, u]:5.1f}%'
        logger.info(res_tmp)
        res_out += res_tmp + '\n'

    # save
    torch.save(net.state_dict(), f'saved/best_edge_task{t}_{args.edge}_{args.dataset}.pt')

for name, metric in zip(['TAw Acc','TAw Forg'], [acc_taw, forg_taw]):
    print('*' * 108)
    logger.info(name)
    for i in range(metric.shape[0]):
        line = '\t'
        for j in range(metric.shape[1]):
            line += f'{100 * metric[i, j]:5.1f}% '
        if np.trace(metric) == 0.0:
            if i > 0:
                line += f'\tAvg.:{100 * metric[i, :i].mean():5.1f}% '
        else:
            line += f'\tAvg.:{100 * metric[i, :i + 1].mean():5.1f}% '
        logger.info(line)
print('*' * 108)

logger.info('[Elapsed time = {:.1f} h]'.format((time.time() - tstart) / (60 * 60)))
logger.info('Done!')