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
from models.spikevgg import SpikeVGG9
from models.vit import VIT

model_conf = {
    'svgg': SpikeVGG9,
    'vit': VIT,
}

def self_logit_distill(outputs, targets, temperature):
        """self-distillation with temperature scaling"""
        new_soft_logits = nn.functional.log_softmax(outputs / temperature, dim=1)
        old_soft_logits = nn.functional.softmax(targets / temperature, dim=1)
        l_old = nn.functional.kl_div(new_soft_logits, old_soft_logits, reduction='batchmean') * (temperature ** 2)

        return l_old

tstart = time.time()
logger = Logger(
    name="update.py", 
    log_file=f"update_{get_local_time()}.log",  # "update.log"
    level=logging.INFO).get_logger()

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
                    default='svgg',
                    type=str,
                    help='edge model name')
parser.add_argument('-cloud',
                    default='vit',
                    type=str,
                    help='cloud model name')
parser.add_argument('-base', 
                    '--nc-first-task', 
                    default=0, 
                    type=int, 
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
parser.add_argument('-l1',
                    type=float,
                    default=0.,
                    help='logit distillation intensity')
parser.add_argument('-pretrain', 
                    action='store_true',
                    help='using pretrained vit model for imagenet')
args = parser.parse_args()
logger.info(args)
seed_all(args.seed)
device = torch.device(f"cuda:{args.gpu_id}" if torch.cuda.is_available() else "cpu")

logger.info(f'Device: {device}')

trn_load = torch.load(f'saved/{args.dataset}/base{args.nc_first_task}_task{args.num_tasks}/train_loader.pt')
tst_load = torch.load(f'saved/{args.dataset}/base{args.nc_first_task}_task{args.num_tasks}/test_loader.pt')
taskcla = torch.load(f'saved/{args.dataset}/base{args.nc_first_task}_task{args.num_tasks}/taskcla.pt')

max_task = len(taskcla) if args.stop_at_task == 0 else args.stop_at_task

# two matrix for final results: task-agnostic
acc_tag = np.zeros((max_task, max_task))
forg_tag = np.zeros((max_task, max_task))

if args.dataset == 'cifar100':
    num_classes = 100
    C, H, W = 3, 32, 32
elif 'imagenet' in args.dataset:
    num_classes = 200
    C, H, W = 3, 64, 64
else:
    raise NotImplementedError(f'Invalid dataset name: {args.dataset}')

# load base edge SNN
init_model = model_conf[args.edge](num_classes, C, H, W, args.T)
seed_all(args.seed)
net = NetHead(init_model)
seed_all(args.seed)

# add base task head
net.add_head(taskcla[0][1]) 
net.set_state_dict(torch.load(f'saved/{args.dataset}/base{args.nc_first_task}_task{args.num_tasks}/best_edge_base_{args.edge}.pt', map_location='cpu'))
net.to(device)

# post process for lwf, preparing for the next task, start from task 1
net_old = deepcopy(net)
net_old.eval()
net_old.freeze_all()

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

class_order = torch.load(f'saved/{args.dataset}/base{args.nc_first_task}_task{args.num_tasks}/classorder.pt')

for t, (_, ncla) in enumerate(taskcla): # task 0->n
    if t >= max_task:
        continue

    print('*' * 108)
    logger.info(f'Task {t:2d}')
    print('*' * 108)

    if t > 0:
        net.add_head(taskcla[t][1]) 
        net.to(device)

        total_cls_in_t = net.task_cls.cumsum(dim=0)
        cloud_label_index = class_order[total_cls_in_t[-2].item():total_cls_in_t[-1].item()]        

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

                # cloud infer
                if args.pretrain:
                    c_output, _ = c_net(
                        nn.functional.interpolate(images.to(device), size=(224, 224), mode='bilinear', align_corners=False))
                else:
                    c_output, _ = c_net(images.to(device))
                # select those labels occured in current task, and convert them to new index
                c_output = c_output[:, cloud_label_index] 

                if args.dataset in ('imagenet'):
                    images = nn.functional.interpolate(images, size=(64, 64), mode='bilinear', align_corners=False)

                outputs_old = None
                if t > 0:
                    outputs_old, _ = net_old(images.to(device))

                outputs, _ = net(images.to(device))

                # L_new
                loss = criterion(outputs[t], targets.to(device) - net.task_offset[t])

                # logit loss, use cloud model to adjust learned label logit distribution
                soft_targets = nn.functional.softmax(c_output / args.temperature, dim=1)
                soft_logits = nn.functional.log_softmax(outputs[t] / args.temperature, dim=1)
                loss_logit = nn.functional.kl_div(soft_logits, soft_targets, reduction='batchmean') * (args.temperature ** 2)
                loss += args.l1 * loss_logit
                
                if t > 0:
                    # L_old
                    loss += args.lamb * self_logit_distill(
                        torch.cat(outputs[:t], dim=1), 
                        torch.cat(outputs_old[:t], dim=1), 
                        args.temperature
                    )

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

            rec_str = f'Epoch {e + 1:3d}, train time={clock1 - clock0:5.1f}s, test time={clock4 - clock3:5.2f}s, loss={test_loss:.3f}, test acc={100 * test_acc:5.2f}%'

            if test_acc >= best_acc:
                best_acc = test_acc
                best_model = net.get_copy()
                patience = args.lr_patience
                rec_str += ' *'
            else:
                patience -= 1
                if patience <= 0:
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
    for u in range(t + 1):
        with torch.no_grad():
            total_acc_tag = 0
            total = 0
            net.eval()
            for images, targets in tst_load[u]:
                if args.dataset in ('imagenet'):
                    images = nn.functional.interpolate(images, size=(64, 64), mode='bilinear', align_corners=False)
                outputs, _ = net(images.to(device))
                # task-agnostic
                pred = torch.cat(outputs, dim=1).argmax(1)
                acc = (pred == targets.to(device)).float()
                total_acc_tag += acc.sum().item()

                total += len(targets)

            test_acc_tag = total_acc_tag / total

        acc_tag[t, u] = test_acc_tag
        if u < t:
            forg_tag[t, u] = acc_tag[:t, u].max(0) - acc_tag[t, u]
        res_tmp = f'>>> Test on task {u:2d} | TAg acc={100 * acc_tag[t, u]:5.1f}%, forg={100 * forg_tag[t, u]:5.1f}%'
        logger.info(res_tmp)

    # save
    torch.save(net.state_dict(), f'saved/{args.dataset}/base{args.nc_first_task}_task{args.num_tasks}/best_edge_task{t}_{args.edge}.pt')

for name, metric in zip(['TAg Acc','TAg Forg'], [acc_tag, forg_tag]):
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

logger.info(f'[Elapsed time = {(time.time() - tstart) / (60 * 60):.1f} h]')
logger.info('Done!')
