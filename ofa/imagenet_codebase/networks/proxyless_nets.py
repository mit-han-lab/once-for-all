# Once for All: Train One Network and Specialize it for Efficient Deployment
# Han Cai, Chuang Gan, Tianzhe Wang, Zhekai Zhang, Song Han
# International Conference on Learning Representations (ICLR), 2020.

import json

import torch.nn as nn
# from layers import *
from ofa.layers import MyModule, ZeroLayer, set_layer_from_config, MBInvertedConvLayer, ConvLayer, IdentityLayer, \
    LinearLayer
from ofa.utils import download_url
from ofa.imagenet_codebase.utils import make_divisible, MyNetwork, int2list


def proxyless_base(net_config=None, n_classes=None, bn_param=None, dropout_rate=None,
                   local_path='~/.torch/proxylessnas/'):
    assert net_config is not None, 'Please input a network config'
    if 'http' in net_config:
        net_config_path = download_url(net_config, local_path)
    else:
        net_config_path = net_config
    net_config_json = json.load(open(net_config_path, 'r'))

    if n_classes is not None:
        net_config_json['classifier']['out_features'] = n_classes
    if dropout_rate is not None:
        net_config_json['classifier']['dropout_rate'] = dropout_rate

    net = ProxylessNASNets.build_from_config(net_config_json)
    if bn_param is not None:
        net.set_bn_param(momentum=bn_param[0], eps=bn_param[1])

    return net


class MobileInvertedResidualBlock(MyModule):

    def __init__(self, mobile_inverted_conv, shortcut):
        super(MobileInvertedResidualBlock, self).__init__()

        self.mobile_inverted_conv = mobile_inverted_conv
        self.shortcut = shortcut

    def forward(self, x):
        if self.mobile_inverted_conv is None or isinstance(self.mobile_inverted_conv, ZeroLayer):
            res = x
        elif self.shortcut is None or isinstance(self.shortcut, ZeroLayer):
            res = self.mobile_inverted_conv(x)
        else:
            res = self.mobile_inverted_conv(x) + self.shortcut(x)
        return res

    @property
    def module_str(self):
        return '(%s, %s)' % (
            self.mobile_inverted_conv.module_str if self.mobile_inverted_conv is not None else None,
            self.shortcut.module_str if self.shortcut is not None else None
        )

    @property
    def config(self):
        return {
            'name': MobileInvertedResidualBlock.__name__,
            'mobile_inverted_conv': self.mobile_inverted_conv.config if self.mobile_inverted_conv is not None else None,
            'shortcut': self.shortcut.config if self.shortcut is not None else None,
        }

    @staticmethod
    def build_from_config(config):
        mobile_inverted_conv = set_layer_from_config(config['mobile_inverted_conv'])
        shortcut = set_layer_from_config(config['shortcut'])
        return MobileInvertedResidualBlock(mobile_inverted_conv, shortcut)


class ProxylessNASNets(MyNetwork):

    def __init__(self, first_conv, blocks, feature_mix_layer, classifier):
        super(ProxylessNASNets, self).__init__()

        self.first_conv = first_conv
        self.blocks = nn.ModuleList(blocks)
        self.feature_mix_layer = feature_mix_layer
        self.classifier = classifier

    def forward(self, x):
        x = self.first_conv(x)
        for block in self.blocks:
            x = block(x)
        if self.feature_mix_layer is not None:
            x = self.feature_mix_layer(x)
        x = x.mean(3).mean(2)
        x = self.classifier(x)
        return x

    @property
    def module_str(self):
        _str = self.first_conv.module_str + '\n'
        for block in self.blocks:
            _str += block.module_str + '\n'
        _str += self.feature_mix_layer.module_str + '\n'
        _str += self.classifier.module_str
        return _str

    @property
    def config(self):
        return {
            'name': ProxylessNASNets.__name__,
            'bn': self.get_bn_param(),
            'first_conv': self.first_conv.config,
            'blocks': [
                block.config for block in self.blocks
            ],
            'feature_mix_layer': None if self.feature_mix_layer is None else self.feature_mix_layer.config,
            'classifier': self.classifier.config,
        }

    @staticmethod
    def build_from_config(config):
        first_conv = set_layer_from_config(config['first_conv'])
        feature_mix_layer = set_layer_from_config(config['feature_mix_layer'])
        classifier = set_layer_from_config(config['classifier'])

        blocks = []
        for block_config in config['blocks']:
            blocks.append(MobileInvertedResidualBlock.build_from_config(block_config))

        net = ProxylessNASNets(first_conv, blocks, feature_mix_layer, classifier)
        if 'bn' in config:
            net.set_bn_param(**config['bn'])
        else:
            net.set_bn_param(momentum=0.1, eps=1e-3)

        return net

    def zero_last_gamma(self):
        for m in self.modules():
            if isinstance(m, MobileInvertedResidualBlock):
                if isinstance(m.mobile_inverted_conv, MBInvertedConvLayer) and isinstance(m.shortcut, IdentityLayer):
                    m.mobile_inverted_conv.point_linear.bn.weight.data.zero_()


class MobileNetV2(ProxylessNASNets):

    def __init__(self, n_classes=1000, width_mult=1, bn_param=(0.1, 1e-3), dropout_rate=0.2,
                 ks=None, expand_ratio=None, depth_param=None, stage_width_list=None):

        if ks is None:
            ks = 3
        if expand_ratio is None:
            expand_ratio = 6

        input_channel = 32
        last_channel = 1280

        input_channel = make_divisible(input_channel * width_mult, 8)
        last_channel = make_divisible(last_channel * width_mult, 8) if width_mult > 1.0 else last_channel

        inverted_residual_setting = [
            # t, c, n, s
            [1, 16, 1, 1],
            [expand_ratio, 24, 2, 2],
            [expand_ratio, 32, 3, 2],
            [expand_ratio, 64, 4, 2],
            [expand_ratio, 96, 3, 1],
            [expand_ratio, 160, 3, 2],
            [expand_ratio, 320, 1, 1],
        ]

        if depth_param is not None:
            assert isinstance(depth_param, int)
            for i in range(1, len(inverted_residual_setting) - 1):
                inverted_residual_setting[i][2] = depth_param

        if stage_width_list is not None:
            for i in range(len(inverted_residual_setting)):
                inverted_residual_setting[i][1] = stage_width_list[i]

        ks = int2list(ks, sum([n for _, _, n, _ in inverted_residual_setting]) - 1)
        _pt = 0

        # first conv layer
        first_conv = ConvLayer(
            3, input_channel, kernel_size=3, stride=2, use_bn=True, act_func='relu6', ops_order='weight_bn_act'
        )
        # inverted residual blocks
        blocks = []
        for t, c, n, s in inverted_residual_setting:
            output_channel = make_divisible(c * width_mult, 8)
            for i in range(n):
                if i == 0:
                    stride = s
                else:
                    stride = 1
                if t == 1:
                    kernel_size = 3
                else:
                    kernel_size = ks[_pt]
                    _pt += 1
                mobile_inverted_conv = MBInvertedConvLayer(
                    in_channels=input_channel, out_channels=output_channel, kernel_size=kernel_size, stride=stride,
                    expand_ratio=t,
                )
                if stride == 1:
                    if input_channel == output_channel:
                        shortcut = IdentityLayer(input_channel, input_channel)
                    else:
                        shortcut = None
                else:
                    shortcut = None
                blocks.append(
                    MobileInvertedResidualBlock(mobile_inverted_conv, shortcut)
                )
                input_channel = output_channel
        # 1x1_conv before global average pooling
        feature_mix_layer = ConvLayer(
            input_channel, last_channel, kernel_size=1, use_bn=True, act_func='relu6', ops_order='weight_bn_act',
        )

        classifier = LinearLayer(last_channel, n_classes, dropout_rate=dropout_rate)

        super(MobileNetV2, self).__init__(first_conv, blocks, feature_mix_layer, classifier)

        # set bn param
        self.set_bn_param(momentum=bn_param[0], eps=bn_param[1])
