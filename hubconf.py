dependencies = ['torch', 'torchvision']

from functools import partial
from ofa.model_zoo import ofa_net, ofa_specialized

# general model
ofa_supernet_resnet50 = partial(ofa_net, net_id="ofa_resnet50", pretrained=True)
ofa_supernet_mbv3_w10 = partial(ofa_net, net_id="ofa_mbv3_d234_e346_k357_w1.0", pretrained=True)
ofa_supernet_mbv3_w12 = partial(ofa_net, net_id="ofa_mbv3_d234_e346_k357_w1.2", pretrained=True)
ofa_supernet_proxyless = partial(ofa_net, net_id="ofa_proxyless_d234_e346_k357_w1.3", pretrained=True)

# specialized
resnet50D_MAC_4_1B = partial(ofa_specialized, net_id="resnet50D_MAC@4.1B_top1@79.8")
resnet50D_MAC_3_7B = partial(ofa_specialized, net_id="resnet50D_MAC@3.7B_top1@79.7")
resnet50D_MAC_3_0B = partial(ofa_specialized, net_id="resnet50D_MAC@3.0B_top1@79.3")
resnet50D_MAC_2_4B = partial(ofa_specialized, net_id="resnet50D_MAC@2.4B_top1@79.0")
resnet50D_MAC_1_8B = partial(ofa_specialized, net_id="resnet50D_MAC@1.8B_top1@78.3")
resnet50D_MAC_1_2B = partial(ofa_specialized, net_id="resnet50D_MAC@1.2B_top1@77.1_finetune@25")
resnet50D_MAC_0_9B = partial(ofa_specialized, net_id="resnet50D_MAC@0.9B_top1@76.3_finetune@25")
resnet50D_MAC_0_6B = partial(ofa_specialized, net_id="resnet50D_MAC@0.6B_top1@75.0_finetune@25")

def ofa_specialized_get():
    return ofa_specialized