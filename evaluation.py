import math
import argparse
import numpy as np

import torch
import torch.nn as nn

import torchvision
import torchvision.transforms as transforms

from utils import *
from models.base import NetHead
from models.spikevgg import SpikeVGG9
from models.vit import VIT

model_conf = {
    'svgg': SpikeVGG9,
    'vit': VIT,
}

# edge latency on Darwin
EDGE_LATENCY = {
    'cifar100-svgg': 1, # TODO
    'imagenet-svgg': 3, # TODO
} 

# cloud latency (ms) = communication latency + computational latency
CLOUD_LATENCY = {
    'cifar100-vit': 1.6 + 4.3,
    'imagenet-vit': 64.4 + 14.1, 
} 

# comunication cost (mJ)
COMMU_COST = {
    'cifar100': 8.77,
    'imagenet': 352.9,
}

# computational cost ()
CLOUD_COST = {
    'cifar100-vit': 290, # TODO
    'imagenet-vit': 290, # TODO
}

EDGE_COST = {
    'cifar100-svgg': 5, # TODO
    'imagenet-svgg': 5, # TODO
}

logger = Logger(
    name="execution.py", 
    log_file="execution.log", # f"execution_{get_local_time()}.log"
    level=logging.INFO).get_logger()

parser = argparse.ArgumentParser(description='Simulate execution stage for ECC-SNN')
parser.add_argument('-seed',
                    default=2025,
                    type=int,
                    help='seed for initializing training.')
parser.add_argument('-gpu',
                    '--gpu_id',
                    default=2,
                    type=int,
                    help='GPU ID to use')
parser.add_argument('-b',
                    '--batch_size',
                    default=64,
                    type=int,
                    metavar='N',
                    help='mini-batch size')
parser.add_argument('-base', 
                    '--nc-first-task', 
                    default=0, 
                    type=int, 
                    required=False,
                    help='Number of classes of the first task')
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
parser.add_argument('-pretrain', 
                    action='store_true',
                    help='using pretrained cloud model for imagenet')
parser.add_argument('-tag', 
                    action='store_true',
                    help='Task-agnostic inference evaluation')
parser.add_argument('-taw', 
                    action='store_true',
                    help='Task-aware inference evaluation')
parser.add_argument('-sensitive', 
                    action='store_true',
                    help='Sensitive analysis for different filter')
parser.add_argument('-thr', 
                    '--threshold', 
                    default=0.7, 
                    type=float, 
                    help='entropy threshold')
args = parser.parse_args()
logger.info(args)
seed_all(args.seed)
device = torch.device(f"cuda:{args.gpu_id}" if torch.cuda.is_available() else "cpu")
logger.info(f'Device: {device}')

taskcla = torch.load(f'saved/{args.dataset}/base{args.nc_first_task}_task{args.num_tasks}/taskcla.pt')
class_order = torch.load(f'saved/{args.dataset}/base{args.nc_first_task}_task{args.num_tasks}/classorder.pt')

if args.dataset == 'cifar100':
    transform_test = transforms.Compose([
        transforms.ToTensor(),
        transforms.Normalize((0.5071, 0.4867, 0.4408), (0.2675, 0.2565, 0.2761)),
    ])
    test_set = torchvision.datasets.CIFAR100(root= '.', train=False, download=False, transform=transform_test)

    num_classes = 100
    C, H, W = 3, 32, 32

elif args.dataset == 'imagenet':
    transform_test = transforms.Compose([
        transforms.Resize(224, interpolation=transforms.InterpolationMode.BICUBIC),
        transforms.ToTensor(),
        transforms.Normalize(mean=(0.485, 0.456, 0.406), std=(0.229, 0.224, 0.225))
    ])

    test_set = TinyImageNetDataset('./tiny-imagenet-200', train=False, transform=transform_test)

    num_classes = 200
    C, H, W = 3, 224, 224
else:
    raise NotImplementedError(f'Invalid dataset name: {args.dataset}')

# evaluate latency and energy consumption
cl = CLOUD_LATENCY[f'{args.dataset}-{args.cloud}'] + EDGE_LATENCY[f'{args.dataset}-{args.edge}']
el = EDGE_LATENCY[f'{args.dataset}-{args.edge}']

ce = CLOUD_COST[f'{args.dataset}-{args.cloud}'] + COMMU_COST[f'{args.dataset}'] + EDGE_COST[f'{args.dataset}-{args.edge}']
ee = EDGE_COST[f'{args.dataset}-{args.edge}']
com_e = COMMU_COST[f'{args.dataset}']

if args.dataset in ('imagenet'):
    # our GPU memory is limited, for SNN model with T=4, it cannot handle 224 input size
    init_model = model_conf[args.edge](num_classes, C, 64, 64, T=args.T)
else:
    init_model = model_conf[args.edge](num_classes, C, H, W, T=args.T)
seed_all(args.seed)
net = NetHead(init_model)
seed_all(args.seed)

# evaluation stop at which task?
max_task = len(taskcla) if args.stop_at_task == 0 else args.stop_at_task

# use the whole test dataset to simulate the open environment and evaluate the edge SNN performance at different task phase
test_loader = torch.utils.data.DataLoader(test_set, batch_size=args.batch_size, shuffle=False)

# initialize the cloud model to assist
if args.pretrain:
    cloud_model = model_conf[args.cloud](num_classes, C, H, W, args.T, args.pretrain)
else:
    cloud_model = model_conf[args.cloud](num_classes, C, H, W, args.T)
cloud_model.load_state_dict(
    torch.load(f'saved/{args.dataset}/base{args.nc_first_task}_task{args.num_tasks}/best_cloud_{args.cloud}.pt', map_location='cpu'))
cloud_model.eval()
cloud_model.to(device)
logger.info('cloud model loaded successfully...')

if args.tag:
    logger.info('Task-Agnostic Simulation Results:')
    for t, (_, ncla) in enumerate(taskcla): # task 0->n
        if t >= max_task:
            continue
        print('*' * 108)
        logger.info(f'Task {t:2d}')
        print('*' * 108)

        net.add_head(taskcla[t][1])
        net.set_state_dict(
            torch.load(f'saved/{args.dataset}/base{args.nc_first_task}_task{args.num_tasks}/best_edge_task{t}_{args.edge}.pt', map_location='cpu'))
        net.to(device)

        entropy_record = []
        label_record = []
        edge_pred_record = []
        cloud_pred_record = []

        with torch.no_grad():
            net.eval()
            for images, targets in test_loader:
                # align with inc tasks
                new_targets = [class_order.index(c.item()) for c in targets]
                label_record.extend(new_targets)

                # cloud infer
                if args.pretrain:
                    cloud_outputs, _ = cloud_model(
                        nn.functional.interpolate(images.to(device), size=(224, 224), mode='bilinear', align_corners=False)
                    )
                else:
                    cloud_outputs, _ = cloud_model(images.to(device))
                _, predicted = torch.max(cloud_outputs.data, 1)
                # reindex cloud infer to class_order index id
                predicted = [class_order.index(c.item()) for c in predicted]
                cloud_pred_record.extend(predicted)

                # edge infer
                if args.dataset in ('imagenet'):
                    edge_outputs, _ = net(
                        nn.functional.interpolate(images.to(device), size=(64, 64), mode='bilinear', align_corners=False)
                    )
                else:
                    edge_outputs, _ = net(images.to(device))
                # edge_outputs = [nn.functional.log_softmax(output, dim=1) for output in edge_outputs]
                edge_pred = torch.cat(edge_outputs, dim=1).argmax(1)
                # calculate entropy for edge outputs
                probs = nn.functional.softmax(torch.cat(edge_outputs, dim=1), dim=-1)
                entropy = -torch.sum(probs * torch.log(probs + 1e-10), dim=-1) / math.log(probs.shape[1])

                entropy_record.extend([c.item() for c in entropy.view(-1)])
                edge_pred_record.extend([c.item() for c in edge_pred.view(-1)])

        import pandas as pd
        print(pd.Series(entropy_record).describe())

        cloud_acc = (np.array(label_record) == np.array(cloud_pred_record)).sum() / len(label_record)
        edge_acc_tag = (np.array(label_record) == np.array(edge_pred_record)).sum() / len(label_record)
        cur = (np.array(entropy_record) > args.threshold).sum() / len(entropy_record)

        avg_energy = (cur * len(entropy_record) * ce + (1 - cur) * len(entropy_record) * ee) / len(entropy_record)
        avg_latency = (cur * len(entropy_record) * cl + (1 - cur) * len(entropy_record) * el) / len(entropy_record)
        avg_com_energy = com_e * cur
        
        logger.info(f'edge acc: {edge_acc_tag * 100:.2f}% cloud acc: {cloud_acc * 100:.2f}% for entire test set')
        logger.info(f'CUR: {cur * 100:.2f}%')
        logger.info(f'avg total energy cost: {avg_energy:.4f}mJ within {avg_com_energy:.4f}mj communication cost, latency {avg_latency:.4f}ms per input')


if args.taw:
    logger.info('Task-Aware Simulation Results:')
    for t, (_, ncla) in enumerate(taskcla): # task 0->n
        if t >= max_task:
            continue
        print('*' * 108)
        logger.info(f'Task {t:2d}')
        print('*' * 108)

        net.add_head(taskcla[t][1])
        net.set_state_dict(
            torch.load(f'saved/{args.dataset}/base{args.nc_first_task}_task{args.num_tasks}/best_edge_task{t}_{args.edge}.pt', map_location='cpu'))
        net.to(device)

        entropy_record = []
        label_record = []
        edge_pred_record = []
        cloud_pred_record = []

        with torch.no_grad():
            net.eval()
            for images, targets in test_loader:
                # align with inc tasks
                new_targets = [class_order.index(c.item()) for c in targets]
                label_record.extend(new_targets)

                # cloud infer
                if args.pretrain:
                    cloud_outputs, _ = cloud_model(
                        nn.functional.interpolate(images.to(device), size=(224, 224), mode='bilinear', align_corners=False)
                    )
                else:
                    cloud_outputs, _ = cloud_model(images.to(device))
                _, predicted = torch.max(cloud_outputs.data, 1)
                predicted = [class_order.index(c.item()) for c in predicted]
                cloud_pred_record.extend(predicted)

                # edge infer
                if args.dataset in ('imagenet'):
                    edge_outputs, _ = net(
                        nn.functional.interpolate(images.to(device), size=(64, 64), mode='bilinear', align_corners=False)
                    )
                else:
                    edge_outputs, _ = net(images.to(device))
                edge_pred = torch.zeros_like(targets.to(device))
                for m in range(len(edge_pred)):
                    if new_targets[m] < net.task_cls.sum().item():
                        this_task = (net.task_cls.cumsum(0) <= new_targets[m]).sum()
                        edge_pred[m] = edge_outputs[this_task][m].argmax() + net.task_offset[this_task]
                    else:
                        edge_pred[m] = num_classes # unknown label for current task

                # edge_outputs = [nn.functional.log_softmax(output, dim=1) for output in edge_outputs]
                probs = nn.functional.softmax(torch.cat(edge_outputs, dim=1), dim=-1)
                entropy = -torch.sum(probs * torch.log(probs + 1e-10), dim=-1) / math.log(probs.shape[1])

                entropy_record.extend([c.item() for c in entropy])
                edge_pred_record.extend([c.item() for c in edge_pred])

        cloud_acc = (np.array(label_record) == np.array(cloud_pred_record)).sum() / len(label_record)
        edge_acc_taw = (np.array(label_record) == np.array(edge_pred_record)).sum() / len(label_record)
        
        cur_flag = np.array(entropy_record) > args.threshold

        ecc_acc_taw = ((np.array(label_record)[cur_flag] == np.array(edge_pred_record)[cur_flag]).sum() + (np.array(label_record)[~cur_flag] == np.array(cloud_pred_record)[~cur_flag]).sum()) / len(label_record)

        cur = cur_flag.sum() / len(entropy_record)
        avg_energy = (cur * len(entropy_record) * ce + (1 - cur) * len(entropy_record) * ee) / len(entropy_record)
        avg_latency = (cur * len(entropy_record) * cl + (1 - cur) * len(entropy_record) * el) / len(entropy_record)
        avg_com_energy = com_e * cur
        
        logger.info(f'edge acc: {edge_acc_taw * 100:.2f}% cloud acc: {cloud_acc * 100:.2f}% for entire test set')
        logger.info(f'ecc-snn acc: {ecc_acc_taw * 100:.2f}% for entire test set')
        logger.info(f'CUR: {cur * 100:.2f}%')
        logger.info(f'avg total energy cost: {avg_energy:.4f}mJ within {avg_com_energy:.4f}mj communication cost, latency {avg_latency:.4f}ms per input')

if args.sensitive:
    logger.info('Sensitive Analysis for CUR v.s. AccI')

    entropy_record = []
    max_p_record = []
    p_margin_record = []
    label_record = []
    edge_pred_record = []
    cloud_pred_record = []
    with torch.no_grad():
        net.eval()
        for images, targets in test_loader:
            new_targets = [class_order.index(c.item()) for c in targets]
            label_record.extend(new_targets)

            edge_outputs, _ = net(images.to(device))

            edge_pred = torch.zeros_like(targets.to(device))
            for m in range(len(edge_pred)):
                if new_targets[m] < net.task_cls.sum().item():
                    this_task = (net.task_cls.cumsum(0) <= new_targets[m]).sum()
                    edge_pred[m] = edge_outputs[this_task][m].argmax() + net.task_offset[this_task]
                else:
                    edge_pred[m] = num_classes # unknown label

            probs = nn.functional.softmax(torch.cat(edge_outputs, dim=1), dim=-1) # (B, D)
            entropy = -torch.sum(probs * torch.log(probs + 1e-10), dim=-1) / math.log(probs.shape[1])

            mp = probs.argmax(dim=-1) # (B)
            max_p_record.extend(list(mp.view(-1)))

            top2_values, _ = torch.topk(probs, k=2, dim=1)
            max_values = top2_values[:, 0]
            second_max_values = top2_values[:, 1]
            p_margin = max_values - second_max_values
            p_margin_record.extend(list(p_margin.view(-1)))

            entropy_record.extend(list(entropy.view(-1)))
            edge_pred_record.extend(list(edge_pred.view(-1)))

            cloud_outputs, _ = cloud_model(images.to(device))
            _, predicted = torch.max(cloud_outputs.data, 1)
            predicted = [class_order.index(c.item()) for c in predicted]
            cloud_pred_record.extend(predicted)

    edge_pred_record = [c.item() for c in edge_pred_record]
    entropy_record = [c.item() for c in entropy_record]
    label_record = [c for c in label_record]
    cloud_pred_record = [c for c in cloud_pred_record]
    p_margin_record = [c.item() for c in p_margin_record]
    max_p_record = [c.item() for c in max_p_record]

    cloud_acc = (np.array(label_record) == np.array(cloud_pred_record)).sum() / len(label_record)
    edge_acc_taw = (np.array(label_record) == np.array(edge_pred_record)).sum() / len(label_record)

    acc_f0 = (np.array(cloud_pred_record) == np.array(label_record)).sum() / len(label_record)
    acc_f1 = (np.array(edge_pred_record) == np.array(label_record)).sum() / len(label_record)

    # # score margin: first-second
    # accis = []
    # curs = []
    # for threshold in range(0, 101, 10):
    #     cur_index = np.array(p_margin_record) < np.percentile(p_margin_record, threshold) 
    #     cur = cur_index.sum() / len(p_margin_record)

    #     acc_c = (np.array(cloud_pred_record)[cur_index] == np.array(label_record)[cur_index]).sum()
    #     acc_e = (np.array(edge_pred_record)[~cur_index] == np.array(label_record)[~cur_index]).sum()
    #     acc_f01 = (acc_c + acc_e) / len(p_margin_record)

    #     accI = (acc_f01 - acc_f1) / (acc_f0 - acc_f1)

    #     print(f'CUR: {int(cur * 100)}%, accI: {accI * 100:.2f}%')
    #     curs.append(int(cur * 100))
    #     accis.append(accI * 100)

    # print('cur:', curs)
    # print('accI:', accis)

    # # max prob
    # accis = []
    # curs = []
    # for threshold in range(0, 101, 10):
    #     cur_index = np.array(max_p_record) < np.percentile(max_p_record, threshold) 
    #     cur = cur_index.sum() / len(max_p_record)

    #     acc_c = (np.array(cloud_pred_record)[cur_index] == np.array(label_record)[cur_index]).sum()
    #     acc_e = (np.array(edge_pred_record)[~cur_index] == np.array(label_record)[~cur_index]).sum()
    #     acc_f01 = (acc_c + acc_e) / len(max_p_record)

    #     accI = (acc_f01 - acc_f1) / (acc_f0 - acc_f1)

    #     print(f'CUR: {int(cur * 100)}%, accI: {accI * 100:.2f}%')
    #     curs.append(int(cur * 100))
    #     accis.append(accI * 100)

    # print('cur:', curs)
    # print('accI:', accis)

    # # entropy
    # accis = []
    # curs = []
    # for threshold in range(0, 101, 10):
    #     cur_index = np.array(entropy_record) >= np.percentile(entropy_record, threshold) 
    #     cur = cur_index.sum() / len(entropy_record)

    #     acc_c = (np.array(cloud_pred_record)[cur_index] == np.array(label_record)[cur_index]).sum()
    #     acc_e = (np.array(edge_pred_record)[~cur_index] == np.array(label_record)[~cur_index]).sum()
    #     acc_f01 = (acc_c + acc_e) / len(entropy_record)

    #     accI = (acc_f01 - acc_f1) / (acc_f0 - acc_f1) 

    #     print(f'CUR: {int(cur * 100)}%, accI: {accI * 100:.2f}%')

    #     curs.append(int(cur * 100))
    #     accis.append(accI * 100)

    # print('entropy')
    # print('cur:', curs)
    # print('accI:', accis)

    # # random
    # accis = []
    # curs = []
    # for threshold in range(0, 101, 10):
    #     selected_index = np.random.choice(np.arange(len(entropy_record)), int(len(entropy_record) * threshold / 100), replace=False) 
    #     cur = len(selected_index) / len(entropy_record)
    #     cur_index = np.isin(np.arange(len(entropy_record)), selected_index)

    #     acc_c = (np.array(cloud_pred_record)[cur_index] == np.array(label_record)[cur_index]).sum()
    #     acc_e = (np.array(edge_pred_record)[~cur_index] == np.array(label_record)[~cur_index]).sum()
    #     acc_f01 = (acc_c + acc_e) / len(entropy_record)

    #     accI = (acc_f01 - acc_f1) / (acc_f0 - acc_f1)

    #     print(f'CUR: {int(cur * 100)}%, accI: {accI * 100:.2f}%')

    #     curs.append(int(cur * 100))
    #     accis.append(accI * 100)

    # print('random')
    # print('cur:', curs)
    # print('accI:', accis)
    ###############################################################################################################


    
