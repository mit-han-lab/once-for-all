dependencies = ['torch', 'torchvision']

from functools import partial
from ofa.model_zoo import ofa_net, ofa_specialized

# general model
ofa_supernet_resnet50 = partial(ofa_net, net_id="ofa_resnet50", pretrained=True)
ofa_supernet_mbv3_w10 = partial(ofa_net, net_id="ofa_mbv3_d234_e346_k357_w1.0", pretrained=True)
ofa_supernet_mbv3_w12 = partial(ofa_net, net_id="ofa_mbv3_d234_e346_k357_w1.2", pretrained=True)
ofa_supernet_proxyless = partial(ofa_net, net_id="ofa_proxyless_d234_e346_k357_w1.3", pretrained=True)

# specialized
resnet50d_mac_06b = partial(ofa_specialized, net_id="resnet50D_MAC@0.6B_top1@75.0_finetune@25", pretrained=True)

