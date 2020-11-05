# Once for All: Train One Network and Specialize it for Efficient Deployment
# Han Cai, Chuang Gan, Tianzhe Wang, Zhekai Zhang, Song Han
# International Conference on Learning Representations (ICLR), 2020.

import os
import os.path as osp
import argparse
import math
from tqdm import tqdm

import torch.nn as nn
import torch.backends.cudnn as cudnn
import torch.utils.data
from torchvision import transforms, datasets

from ofa.utils import AverageMeter, accuracy
from ofa.model_zoo import ofa_specialized

specialized_network_list = [
    ################# FLOPs #################
    'flops@595M_top1@80.0_finetune@75',
    'flops@482M_top1@79.6_finetune@75',
    'flops@389M_top1@79.1_finetune@75',
    ################# ResNet50 Design Space #################
    'resnet50D_MAC@4.1B_top1@79.8',
    'resnet50D_MAC@3.7B_top1@79.7',
    'resnet50D_MAC@3.0B_top1@79.3',
    'resnet50D_MAC@2.4B_top1@79.0',
    'resnet50D_MAC@1.8B_top1@78.3',
    'resnet50D_MAC@1.2B_top1@77.1_finetune@25',
    'resnet50D_MAC@0.9B_top1@76.3_finetune@25',
    'resnet50D_MAC@0.6B_top1@75.0_finetune@25',
    ################# Google pixel1 #################
    'pixel1_lat@143ms_top1@80.1_finetune@75',
    'pixel1_lat@132ms_top1@79.8_finetune@75',
    'pixel1_lat@79ms_top1@78.7_finetune@75',
    'pixel1_lat@58ms_top1@76.9_finetune@75',
    'pixel1_lat@40ms_top1@74.9_finetune@25',
    'pixel1_lat@28ms_top1@73.3_finetune@25',
    'pixel1_lat@20ms_top1@71.4_finetune@25',
    ################# Google pixel2 #################
    'pixel2_lat@62ms_top1@75.8_finetune@25',
    'pixel2_lat@50ms_top1@74.7_finetune@25',
    'pixel2_lat@35ms_top1@73.4_finetune@25',
    'pixel2_lat@25ms_top1@71.5_finetune@25',
    ################# Samsung note10 #################
    'note10_lat@64ms_top1@80.2_finetune@75',
    'note10_lat@50ms_top1@79.7_finetune@75',
    'note10_lat@41ms_top1@79.3_finetune@75',
    'note10_lat@30ms_top1@78.4_finetune@75',
    'note10_lat@22ms_top1@76.6_finetune@25',
    'note10_lat@16ms_top1@75.5_finetune@25',
    'note10_lat@11ms_top1@73.6_finetune@25',
    'note10_lat@8ms_top1@71.4_finetune@25',
    ################# Samsung note8 #################
    'note8_lat@65ms_top1@76.1_finetune@25',
    'note8_lat@49ms_top1@74.9_finetune@25',
    'note8_lat@31ms_top1@72.8_finetune@25',
    'note8_lat@22ms_top1@70.4_finetune@25',
    ################# Samsung S7 Edge #################
    's7edge_lat@88ms_top1@76.3_finetune@25',
    's7edge_lat@58ms_top1@74.7_finetune@25',
    's7edge_lat@41ms_top1@73.1_finetune@25',
    's7edge_lat@29ms_top1@70.5_finetune@25',
    ################# LG G8 #################
    'LG-G8_lat@24ms_top1@76.4_finetune@25',
    'LG-G8_lat@16ms_top1@74.7_finetune@25',
    'LG-G8_lat@11ms_top1@73.0_finetune@25',
    'LG-G8_lat@8ms_top1@71.1_finetune@25',
    ################# 1080ti GPU (Batch Size 64) #################
    '1080ti_gpu64@27ms_top1@76.4_finetune@25',
    '1080ti_gpu64@22ms_top1@75.3_finetune@25',
    '1080ti_gpu64@15ms_top1@73.8_finetune@25',
    '1080ti_gpu64@12ms_top1@72.6_finetune@25',
    ################# V100 GPU (Batch Size 64) #################
    'v100_gpu64@11ms_top1@76.1_finetune@25',
    'v100_gpu64@9ms_top1@75.3_finetune@25',
    'v100_gpu64@6ms_top1@73.0_finetune@25',
    'v100_gpu64@5ms_top1@71.6_finetune@25',
    ################# Jetson TX2 GPU (Batch Size 16) #################
    'tx2_gpu16@96ms_top1@75.8_finetune@25',
    'tx2_gpu16@80ms_top1@75.4_finetune@25',
    'tx2_gpu16@47ms_top1@72.9_finetune@25',
    'tx2_gpu16@35ms_top1@70.3_finetune@25',
    ################# Intel Xeon CPU with MKL-DNN (Batch Size 1) #################
    'cpu_lat@17ms_top1@75.7_finetune@25',
    'cpu_lat@15ms_top1@74.6_finetune@25',
    'cpu_lat@11ms_top1@72.0_finetune@25',
    'cpu_lat@10ms_top1@71.1_finetune@25',
]

parser = argparse.ArgumentParser()
parser.add_argument(
    '-p',
    '--path',
    help='The path of imagenet',
    type=str,
    default='/dataset/imagenet')
parser.add_argument(
    '-g',
    '--gpu',
    help='The gpu(s) to use',
    type=str,
    default='all')
parser.add_argument(
    '-b',
    '--batch-size',
    help='The batch on every device for validation',
    type=int,
    default=100)
parser.add_argument(
    '-j',
    '--workers',
    help='Number of workers',
    type=int,
    default=20)
parser.add_argument(
    '-n',
    '--net',
    metavar='NET',
    default='pixel1_lat@143ms_top1@80.1_finetune@75',
    choices=specialized_network_list,
    help='OFA specialized networks: ' +
    ' | '.join(specialized_network_list) +
    ' (default: pixel1_lat@143ms_top1@80.1_finetune@75)')

args = parser.parse_args()
if args.gpu == 'all':
    device_list = range(torch.cuda.device_count())
    args.gpu = ','.join(str(_) for _ in device_list)
else:
    device_list = [int(_) for _ in args.gpu.split(',')]
os.environ['CUDA_VISIBLE_DEVICES'] = args.gpu

net, image_size = ofa_specialized(net_id=args.net, pretrained=True)
args.batch_size = args.batch_size * max(len(device_list), 1)

data_loader = torch.utils.data.DataLoader(
    datasets.ImageFolder(
        osp.join(
            args.path,
            'val'),
        transforms.Compose(
            [
                transforms.Resize(int(math.ceil(image_size / 0.875))),
                transforms.CenterCrop(image_size),
                transforms.ToTensor(),
                transforms.Normalize(
                    mean=[
                        0.485,
                        0.456,
                        0.406],
                    std=[
                        0.229,
                        0.224,
                        0.225]),
            ])),
    batch_size=args.batch_size,
    shuffle=True,
    num_workers=args.workers,
    pin_memory=True,
    drop_last=False,
)

net = torch.nn.DataParallel(net).cuda()
cudnn.benchmark = True
criterion = nn.CrossEntropyLoss().cuda()

net.eval()
losses = AverageMeter()
top1 = AverageMeter()
top5 = AverageMeter()

with torch.no_grad():
    with tqdm(total=len(data_loader), desc='Validate') as t:
        for i, (images, labels) in enumerate(data_loader):
            images, labels = images.cuda(), labels.cuda()
            # compute output
            output = net(images)
            loss = criterion(output, labels)
            # measure accuracy and record loss
            acc1, acc5 = accuracy(output, labels, topk=(1, 5))

            losses.update(loss.item(), images.size(0))
            top1.update(acc1[0].item(), images.size(0))
            top5.update(acc5[0].item(), images.size(0))
            t.set_postfix({
                'loss': losses.avg,
                'top1': top1.avg,
                'top5': top5.avg,
                'img_size': images.size(2),
            })
            t.update(1)

print('Test OFA specialized net <%s> with image size %d:' % (args.net, image_size))
print('Results: loss=%.5f,\t top1=%.1f,\t top5=%.1f' % (losses.avg, top1.avg, top5.avg))
