import os
import torch
import torch.nn as nn
from torchvision import transforms, datasets
import numpy as np
import time
import random
import math
import copy
from matplotlib import pyplot as plt

from ofa.model_zoo import ofa_net
from ofa.utils import download_url

# from ofa.tutorial.accuracy_predictor import AccuracyPredictor
# from ofa.tutorial.flops_table import FLOPsTable
# from ofa.tutorial.latency_table import LatencyTable
# from ofa.tutorial.evolution_finder import EvolutionFinder
# from ofa.tutorial.imagenet_eval_helper import evaluate_ofa_subnet, evaluate_ofa_specialized
from ofa.tutorial import AccuracyPredictor, FLOPsTable, LatencyTable, EvolutionFinder
from ofa.tutorial import evaluate_ofa_subnet, evaluate_ofa_specialized

# set random seed
random_seed = 1
random.seed(random_seed)
np.random.seed(random_seed)
torch.manual_seed(random_seed)
print('Successfully imported all packages and configured random seed to %d!'%random_seed)

#os.environ['CUDA_VISIBLE_DEVICES'] = '0'
cuda_available = torch.cuda.is_available()
if cuda_available:
    torch.backends.cudnn.enabled = True
    torch.backends.cudnn.benchmark = True
    torch.cuda.manual_seed(random_seed)
    print('Using GPU.')
else:
    print('Using CPU.')

ofa_network = ofa_net('ofa_mbv3_d234_e346_k357_w1.2', pretrained=True)
print('The OFA Network is ready.')

imagenet_data_path = 'imagenet_1k'

if cuda_available:
    # The following function build the data transforms for test
    def build_val_transform(size):
        return transforms.Compose([
            transforms.Resize(int(math.ceil(size / 0.875))),
            transforms.CenterCrop(size),
            transforms.ToTensor(),
            transforms.Normalize(
                mean=[0.485, 0.456, 0.406],
                std=[0.229, 0.224, 0.225]
            ),
        ])

    data_loader = torch.utils.data.DataLoader(
        datasets.ImageFolder(
            root=os.path.join(imagenet_data_path, 'val'),
            transform=build_val_transform(224)
        ),
        batch_size=250,  # test batch size
        shuffle=True,
        num_workers=16,  # number of workers for the data loader
        pin_memory=True,
        drop_last=False,
    )
    print('The ImageNet dataloader is ready.')
else:
    data_loader = None
    print('Since GPU is not found in the environment, we skip all scripts related to ImageNet evaluation.')

if cuda_available:
    net_id = evaluate_ofa_specialized(imagenet_data_path, data_loader)
    print('Finished evaluating the pretrained sub-network: %s!' % net_id)
else:
    print('Since GPU is not found in the environment, we skip all scripts related to ImageNet evaluation.')

accuracy_predictor = AccuracyPredictor(
    pretrained=True,
    device='cuda:0' if cuda_available else 'cpu'
)

print('The accuracy predictor is ready!')
print(accuracy_predictor.model)

target_hardware = 'note10'
latency_table = LatencyTable(device=target_hardware)
print('The Latency lookup table on %s is ready!' % target_hardware)


""" Hyper-parameters for the evolutionary search process
    You can modify these hyper-parameters to see how they influence the final ImageNet accuracy of the search sub-net.
"""
latency_constraint = 25  # ms, suggested range [15, 33] ms
P = 100  # The size of population in each generation
N = 500  # How many generations of population to be searched
r = 0.25  # The ratio of networks that are used as parents for next generation
params = {
    'constraint_type': target_hardware, # Let's do FLOPs-constrained search
    'efficiency_constraint': latency_constraint,
    'mutate_prob': 0.1, # The probability of mutation in evolutionary search
    'mutation_ratio': 0.5, # The ratio of networks that are generated through mutation in generation n >= 2.
    'efficiency_predictor': latency_table, # To use a predefined efficiency predictor.
    'accuracy_predictor': accuracy_predictor, # To use a predefined accuracy_predictor predictor.
    'population_size': P,
    'max_time_budget': N,
    'parent_ratio': r,
}

# build the evolution finder
finder = EvolutionFinder(**params)

# start searching
result_lis = []
st = time.time()
best_valids, best_info = finder.run_evolution_search()
result_lis.append(best_info)
ed = time.time()
print('Found best architecture on %s with latency <= %.2f ms in %.2f seconds! '
      'It achieves %.2f%s predicted accuracy with %.2f ms latency on %s.' %
      (target_hardware, latency_constraint, ed-st, best_info[0] * 100, '%', best_info[-1], target_hardware))

# visualize the architecture of the searched sub-net
_, net_config, latency = best_info
ofa_network.set_active_subnet(ks=net_config['ks'], d=net_config['d'], e=net_config['e'])
print('Architecture of the searched sub-net:')
print(ofa_network.module_str)

# evaluate the searched model on ImageNet
if cuda_available:
    top1s = []
    latency_list = []
    for result in result_lis:
        _, net_config, latency = result
        print('Evaluating the sub-network with latency = %.1f ms on %s' % (latency, target_hardware))
        top1 = evaluate_ofa_subnet(
            ofa_network,
            imagenet_data_path,
            net_config,
            data_loader,
            batch_size=250,
            device='cuda:0' if cuda_available else 'cpu')
        top1s.append(top1)
        latency_list.append(latency)

    plt.figure(figsize=(4,4))
    plt.plot(latency_list, top1s, 'x-', marker='*', color='darkred',  linewidth=2, markersize=8, label='OFA')
    plt.plot([26, 45], [74.6, 76.7], '--', marker='+', linewidth=2, markersize=8, label='ProxylessNAS')
    plt.plot([15.3, 22, 31], [73.3, 75.2, 76.6], '--', marker='>', linewidth=2, markersize=8, label='MobileNetV3')
    plt.xlabel('%s Latency (ms)' % target_hardware, size=12)
    plt.ylabel('ImageNet Top-1 Accuracy (%)', size=12)
    plt.legend(['OFA', 'ProxylessNAS', 'MobileNetV3'], loc='lower right')
    plt.grid(True)
    plt.show()
    print('Successfully draw the tradeoff curve!')
else:
    print('Since GPU is not found in the environment, we skip all scripts related to ImageNet evaluation.')


flops_lookup_table = FLOPsTable(
    device='cuda:0' if cuda_available else 'cpu',
    batch_size=1,
)
print('The FLOPs lookup table is ready!')


""" Hyper-parameters for the evolutionary search process
    You can modify these hyper-parameters to see how they influence the final ImageNet accuracy of the search sub-net.
"""
P = 100  # The size of population in each generation
N = 500  # How many generations of population to be searched
r = 0.25  # The ratio of networks that are used as parents for next generation
params = {
    'constraint_type': 'flops', # Let's do FLOPs-constrained search
    'efficiency_constraint': 600,  # FLops constraint (M), suggested range [150, 600]
    'mutate_prob': 0.1, # The probability of mutation in evolutionary search
    'mutation_ratio': 0.5, # The ratio of networks that are generated through mutation in generation n >= 2.
    'efficiency_predictor': flops_lookup_table, # To use a predefined efficiency predictor.
    'accuracy_predictor': accuracy_predictor, # To use a predefined accuracy_predictor predictor.
    'population_size': P,
    'max_time_budget': N,
    'parent_ratio': r,
}

# build the evolution finder
finder = EvolutionFinder(**params)

# start searching
result_lis = []
for flops in [600, 400, 350]:
    st = time.time()
    finder.set_efficiency_constraint(flops)
    best_valids, best_info = finder.run_evolution_search()
    ed = time.time()
    # print('Found best architecture at flops <= %.2f M in %.2f seconds! It achieves %.2f%s predicted accuracy with %.2f MFLOPs.' % (flops, ed-st, best_info[0] * 100, '%', best_info[-1]))
    result_lis.append(best_info)

plt.figure(figsize=(4,4))
plt.plot([x[-1] for x in result_lis], [x[0] * 100 for x in result_lis], 'x-', marker='*', color='darkred',  linewidth=2, markersize=8, label='OFA')
plt.xlabel('FLOPs (M)', size=12)
plt.ylabel('Predicted Holdout Top-1 Accuracy (%)', size=12)
plt.legend(['OFA'], loc='lower right')
plt.grid(True)
plt.show()


if cuda_available:
    # test the searched model on the test dataset (ImageNet val)
    top1s = []
    flops_lis = []
    for result in result_lis:
        _, net_config, flops = result
        print('Evaluating the sub-network with FLOPs = %.1fM' % flops)
        top1 = evaluate_ofa_subnet(
            ofa_network,
            imagenet_data_path,
            net_config,
            data_loader,
            batch_size=250,
            device='cuda:0' if cuda_available else 'cpu')
        print('-' * 45)
        top1s.append(top1)
        flops_lis.append(flops)

    plt.figure(figsize=(8,4))
    plt.subplot(1, 2, 1)
    plt.plot([x[-1] for x in result_lis], [x[0] * 100 for x in result_lis], 'x-', marker='*', color='darkred',  linewidth=2, markersize=8, label='OFA')
    plt.xlabel('FLOPs (M)', size=12)
    plt.ylabel('Predicted Holdout Top-1 Accuracy (%)', size=12)
    plt.legend(['OFA'], loc='lower right')
    plt.grid(True)

    plt.subplot(1, 2, 2)
    plt.plot(flops_lis, top1s, 'x-', marker='*', color='darkred',  linewidth=2, markersize=8, label='OFA')
    plt.plot([320, 581], [74.6, 76.7], '--', marker='+', linewidth=2, markersize=8, label='ProxylessNAS')
    plt.plot([219, 343], [75.2, 76.6], '--', marker='^', linewidth=2, markersize=8, label='MobileNetV3')
    plt.plot([390, 700], [76.3, 78.8], '--', marker='>', linewidth=2, markersize=8, label='EfficientNet')
    plt.xlabel('FLOPs (M)', size=12)
    plt.ylabel('ImageNet Top-1 Accuracy (%)', size=12)
    plt.legend(['OFA', 'ProxylessNAS', 'MobileNetV3', 'EfficientNet'], loc='lower right')
    plt.grid(True)
    plt.show()

