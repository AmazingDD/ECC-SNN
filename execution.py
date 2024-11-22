import math
import argparse
import numpy as np

import torch
import torch.nn as nn

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

# inference latency (ms), see scripts/ for more details TODO
CLOUD_LATENCY = {
    'cifar100-vgg16': 290,
    'cifar100-vit4': 290, # TODO
} 
EDGE_LATENCY = {
    'cifar100-svgg9': 35, # TODO
    'cifar100-svit1': 290, # TODO
} 

# energy (mJ) TODO calculate with https://github.com/TWTcodeKing/energy_sim
CLOUD_ENERGY = {
    'cifar100-vgg16': 100, # TODO
    'cifar100-vit4': 290, # TODO
}
EDGE_ENERGY = {
    'cifar100-svgg9': 5, # TODO
    'cifar100-svit1': 5, # TODO
}

logger = Logger(name="execution.py", log_file="execution.log", level=logging.INFO).get_logger()

parser = argparse.ArgumentParser(description='Simulate execution stage for ECC-SNN')
parser.add_argument('-seed',
                    default=2025,
                    type=int,
                    help='seed for initializing training.')
parser.add_argument('-gpu',
                    '--gpu_id',
                    default=6,
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
                    default=None, 
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
                    default='svgg9',
                    type=str,
                    help='edge model name')
parser.add_argument('-cloud',
                    default='vgg16',
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
args = parser.parse_args()
logger.info(args)
seed_all(args.seed)
device = torch.device(f"cuda:{args.gpu_id}" if torch.cuda.is_available() else "cpu")

taskcla = torch.load(f'saved/{args.dataset}/base{args.nc_first_task}_task{args.num_tasks}/taskcla.pt')
class_order = torch.load(f'saved/{args.dataset}/base{args.nc_first_task}_task{args.num_tasks}/classorder.pt')
tst_load = torch.load(f'saved/{args.dataset}/base{args.nc_first_task}_task{args.num_tasks}/test_loader.pt')

if args.dataset == 'cifar10':
    transform_test = transforms.Compose([
        transforms.ToTensor(),
        transforms.Normalize((0.4914, 0.4822, 0.4465), (0.2023, 0.1994, 0.2010)),
    ])
    test_set = torchvision.datasets.CIFAR10(root= '.', train=False, download=False, transform=transform_test)

    num_classes = 10
    C, H, W = 3, 32, 32

elif args.dataset == 'cifar100':
    transform_test = transforms.Compose([
        transforms.ToTensor(),
        transforms.Normalize((0.5071, 0.4867, 0.4408), (0.2675, 0.2565, 0.2761)),
    ])
    test_set = torchvision.datasets.CIFAR100(root= '.', train=False, download=False, transform=transform_test)

    num_classes = 100
    C, H, W = 3, 32, 32

elif args.dataset == 'tiny-imagenet':
    # TODO
    num_classes = 200
    C, H, W = 3, 128, 128
else:
    raise NotImplementedError(f'Invalid dataset name: {args.dataset}')

# evaluate latency and energy consumption
cl = CLOUD_LATENCY[f'{args.dataset}-{args.cloud}']
el = EDGE_LATENCY[f'{args.dataset}-{args.edge}']

ce = CLOUD_ENERGY[f'{args.dataset}-{args.cloud}']
ee = EDGE_ENERGY[f'{args.dataset}-{args.edge}']

init_model = model_conf[args.edge](num_classes, C, H, W)
init_model.T = args.T
seed_all(args.seed)
net = NetHead(init_model)
seed_all(args.seed)

max_task = len(taskcla) if args.stop_at_task == 0 else args.stop_at_task

test_loader = torch.utils.data.DataLoader(test_set, batch_size=args.batch_size, shuffle=False)

cloud_model = model_conf[args.cloud](num_classes, C, H, W)
cloud_model.load_state_dict(
    torch.load(f'saved/{args.dataset}/base{args.nc_first_task}_task{args.num_tasks}/best_cloud_{args.cloud}.pt', map_location='cpu'))
cloud_model.eval()
cloud_model.to(device)

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

    ################################################ CUR v.s. AccI ################################################
    # entropy_record = []
    # max_p_record = []
    # p_margin_record = []
    # label_record = []
    # edge_pred_record = []
    # cloud_pred_record = []
    # with torch.no_grad():
    #     net.eval()
    #     for images, targets in test_loader:
    #         new_targets = [class_order.index(c.item()) for c in targets]
    #         label_record.extend(new_targets)

    #         edge_outputs, _ = net(images.to(device))

    #         edge_pred = torch.zeros_like(targets.to(device))
    #         for m in range(len(edge_pred)):
    #             if new_targets[m] < net.task_cls.sum().item():
    #                 this_task = (net.task_cls.cumsum(0) <= new_targets[m]).sum()
    #                 edge_pred[m] = edge_outputs[this_task][m].argmax() + net.task_offset[this_task]
    #             else:
    #                 edge_pred[m] = num_classes # unknown label

    #         probs = nn.functional.softmax(torch.cat(edge_outputs, dim=1), dim=-1) # (B, D)
    #         entropy = -torch.sum(probs * torch.log(probs + 1e-10), dim=-1) / math.log(probs.shape[1])

    #         mp = probs.argmax(dim=-1) # (B)
    #         max_p_record.extend(list(mp.view(-1)))

    #         top2_values, _ = torch.topk(probs, k=2, dim=1)
    #         max_values = top2_values[:, 0]
    #         second_max_values = top2_values[:, 1]
    #         p_margin = max_values - second_max_values
    #         p_margin_record.extend(list(p_margin.view(-1)))

    #         entropy_record.extend(list(entropy.view(-1)))
    #         edge_pred_record.extend(list(edge_pred.view(-1)))

    #         cloud_outputs, _ = cloud_model(images.to(device))
    #         _, predicted = torch.max(cloud_outputs.data, 1)
    #         predicted = [class_order.index(c.item()) for c in predicted]
    #         cloud_pred_record.extend(predicted)

    # edge_pred_record = [c.item() for c in edge_pred_record]
    # entropy_record = [c.item() for c in entropy_record]
    # label_record = [c for c in label_record]
    # cloud_pred_record = [c for c in cloud_pred_record]
    # p_margin_record = [c.item() for c in p_margin_record]
    # max_p_record = [c.item() for c in max_p_record]

    # cloud_acc = (np.array(label_record) == np.array(cloud_pred_record)).sum() / len(label_record)
    # edge_acc_taw = (np.array(label_record) == np.array(edge_pred_record)).sum() / len(label_record)

    # acc_f0 = (np.array(cloud_pred_record) == np.array(label_record)).sum() / len(label_record)
    # acc_f1 = (np.array(edge_pred_record) == np.array(label_record)).sum() / len(label_record)

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

    
    ################################# task agnostic evaluation on entire test dataset #############################
    # entropy_record = []
    # label_record = []
    # edge_pred_record = []
    # cloud_pred_record = []

    # with torch.no_grad():
    #     net.eval()
    #     for images, targets in test_loader:
    #         # align with inc tasks
    #         new_targets = [class_order.index(c.item()) for c in targets]
    #         label_record.extend(new_targets)

    #         edge_outputs, _ = net(images.to(device))
    #         # edge_outputs = [nn.functional.log_softmax(output, dim=1) for output in edge_outputs]
    #         edge_pred = torch.cat(edge_outputs, dim=1).argmax(1)
    #         probs = nn.functional.softmax(torch.cat(edge_outputs, dim=1), dim=-1)
    #         entropy = -torch.sum(probs * torch.log(probs + 1e-10), dim=-1) / math.log(probs.shape[1])

    #         entropy_record.extend(list(entropy.view(-1)))
    #         edge_pred_record.extend(list(edge_pred.view(-1)))

    #         cloud_outputs, _ = cloud_model(images.to(device))
    #         _, predicted = torch.max(cloud_outputs.data, 1)
    #         predicted = [class_order.index(c.item()) for c in predicted]
    #         cloud_pred_record.extend(predicted)

    # edge_pred_record = [c.item() for c in edge_pred_record]
    # entropy_record = [c.item() for c in entropy_record]
    # label_record = [c for c in label_record]
    # cloud_pred_record = [c for c in cloud_pred_record]

    # cloud_acc = (np.array(label_record) == np.array(cloud_pred_record)).sum() / len(label_record)
    # edge_acc_tag = (np.array(label_record) == np.array(edge_pred_record)).sum() / len(label_record)
    # cur = (np.array(entropy_record) > 0.7).sum() / len(entropy_record)
    
    # print(f'edge acc: {edge_acc_tag * 100:.2f}% cloud acc: {cloud_acc * 100:.2f}% for entire test set (Task-agnostic)')
    # print(f'CUR: {cur * 100:.2f}%')
    ###################################################################################################################


    ######################## task-aware evaluation on entire test dataset with evolving rounds ########################
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

            edge_outputs, _ = net(images.to(device))

            edge_pred = torch.zeros_like(targets.to(device))
            for m in range(len(edge_pred)):
                if new_targets[m] < net.task_cls.sum().item():
                    this_task = (net.task_cls.cumsum(0) <= new_targets[m]).sum()
                    edge_pred[m] = edge_outputs[this_task][m].argmax() + net.task_offset[this_task]
                else:
                    edge_pred[m] = num_classes # unknown label

            # edge_outputs = [nn.functional.log_softmax(output, dim=1) for output in edge_outputs]
            probs = nn.functional.softmax(torch.cat(edge_outputs, dim=1), dim=-1)
            entropy = -torch.sum(probs * torch.log(probs + 1e-10), dim=-1) / math.log(probs.shape[1])

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

    cloud_acc = (np.array(label_record) == np.array(cloud_pred_record)).sum() / len(label_record)
    edge_acc_taw = (np.array(label_record) == np.array(edge_pred_record)).sum() / len(label_record)
    cur = (np.array(entropy_record) > 0.7).sum() / len(entropy_record)

    avg_energy = (cur * len(entropy_record) * ce + (1 - cur) * len(entropy_record) * ee) / len(entropy_record)
    avg_latency = (cur * len(entropy_record) * cl + (1 - cur) * len(entropy_record) * el) / len(entropy_record)
    
    print(f'edge acc: {edge_acc_taw * 100:.2f}% cloud acc: {cloud_acc * 100:.2f}% for entire test set (Task-aware)')
    print(f'CUR: {cur * 100:.2f}%')
    print(f'avg energy cost: {avg_energy:.4f}mJ, latency {avg_latency:.4f}ms per input')
    ###################################################################################################################
