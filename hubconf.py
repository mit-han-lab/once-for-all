dependencies = ['torch', 'torchvision']

from model_zoo import ofa_net, ofa_specialized

# general model
ofa_resnet50 = ofa_net("ofa_resnet50", pretrained=True)
ofa_mbv3_w10 = ofa_net("ofa_mbv3_d234_e346_k357_w1.0", pretrained=True)
ofa_mbv3_w12 = ofa_net("ofa_mbv3_d234_e346_k357_w1.2", pretrained=True)
ofa_proxyless = ofa_net("ofa_proxyless_d234_e346_k357_w1.3", pretrained=True)


# specialized
resnet50d_mac_06b = ofa_specialized("resnet50D_MAC@0.6B_top1@75.0_finetune@25", pretrained=True)

