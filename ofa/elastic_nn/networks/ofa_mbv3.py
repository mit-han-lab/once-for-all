# Once for All: Train One Network and Specialize it for Efficient Deployment
# Han Cai, Chuang Gan, Tianzhe Wang, Zhekai Zhang, Song Han
# International Conference on Learning Representations (ICLR), 2020.

import copy
import random

import torch

from ofa.elastic_nn.modules.dynamic_layers import DynamicMBConvLayer, DynamicConvLayer, DynamicLinearLayer
from ofa.layers import ConvLayer, IdentityLayer, LinearLayer, MBInvertedConvLayer
from ofa.imagenet_codebase.networks.mobilenet_v3 import MobileNetV3, MobileInvertedResidualBlock
from ofa.imagenet_codebase.utils import make_divisible, int2list


class OFAMobileNetV3(MobileNetV3):

    def __init__(self, n_classes=1000, bn_param=(0.1, 1e-5), dropout_rate=0.1, base_stage_width=None,
                 width_mult_list=1.0, ks_list=3, expand_ratio_list=6, depth_list=4):

        self.width_mult_list = int2list(width_mult_list, 1)
        self.ks_list = int2list(ks_list, 1)
        self.expand_ratio_list = int2list(expand_ratio_list, 1)
        self.depth_list = int2list(depth_list, 1)
        self.base_stage_width = base_stage_width

        self.width_mult_list.sort()
        self.ks_list.sort()
        self.expand_ratio_list.sort()
        self.depth_list.sort()

        base_stage_width = [16, 24, 40, 80, 112, 160, 960, 1280]

        final_expand_width = [
            make_divisible(base_stage_width[-2] * max(self.width_mult_list), 8) for _ in self.width_mult_list
        ]
        last_channel = [
            make_divisible(base_stage_width[-1] * max(self.width_mult_list), 8) for _ in self.width_mult_list
        ]

        stride_stages = [1, 2, 2, 2, 1, 2]
        act_stages = ['relu', 'relu', 'relu', 'h_swish', 'h_swish', 'h_swish']
        se_stages = [False, False, True, False, True, True]
        if depth_list is None:
            n_block_list = [1, 2, 3, 4, 2, 3]
            self.depth_list = [4, 4]
            print('Use MobileNetV3 Depth Setting')
        else:
            n_block_list = [1] + [max(self.depth_list)] * 5
        width_list = []
        for base_width in base_stage_width[:-2]:
            width = [make_divisible(base_width * width_mult, 8) for width_mult in self.width_mult_list]
            width_list.append(width)

        input_channel = width_list[0]
        # first conv layer
        if len(set(input_channel)) == 1:
            first_conv = ConvLayer(3, max(input_channel), kernel_size=3, stride=2, act_func='h_swish')
            first_block_conv = MBInvertedConvLayer(
                in_channels=max(input_channel), out_channels=max(input_channel), kernel_size=3, stride=stride_stages[0],
                expand_ratio=1, act_func=act_stages[0], use_se=se_stages[0],
            )
        else:
            first_conv = DynamicConvLayer(
                in_channel_list=int2list(3, len(input_channel)), out_channel_list=input_channel, kernel_size=3,
                stride=2, act_func='h_swish',
            )
            first_block_conv = DynamicMBConvLayer(
                in_channel_list=input_channel, out_channel_list=input_channel, kernel_size_list=3, expand_ratio_list=1,
                stride=stride_stages[0], act_func=act_stages[0], use_se=se_stages[0],
            )
        first_block = MobileInvertedResidualBlock(first_block_conv, IdentityLayer(input_channel, input_channel))

        # inverted residual blocks
        self.block_group_info = []
        blocks = [first_block]
        _block_index = 1
        feature_dim = input_channel

        for width, n_block, s, act_func, use_se in zip(width_list[1:], n_block_list[1:],
                                                       stride_stages[1:], act_stages[1:], se_stages[1:]):
            self.block_group_info.append([_block_index + i for i in range(n_block)])
            _block_index += n_block

            output_channel = width
            for i in range(n_block):
                if i == 0:
                    stride = s
                else:
                    stride = 1
                mobile_inverted_conv = DynamicMBConvLayer(
                    in_channel_list=feature_dim, out_channel_list=output_channel, kernel_size_list=ks_list,
                    expand_ratio_list=expand_ratio_list, stride=stride, act_func=act_func, use_se=use_se,
                )
                if stride == 1 and feature_dim == output_channel:
                    shortcut = IdentityLayer(feature_dim, feature_dim)
                else:
                    shortcut = None
                blocks.append(MobileInvertedResidualBlock(mobile_inverted_conv, shortcut))
                feature_dim = output_channel
        # final expand layer, feature mix layer & classifier
        if len(final_expand_width) == 1:
            final_expand_layer = ConvLayer(max(feature_dim), max(final_expand_width), kernel_size=1, act_func='h_swish')
            feature_mix_layer = ConvLayer(
                max(final_expand_width), max(last_channel), kernel_size=1, bias=False, use_bn=False, act_func='h_swish',
            )
        else:
            final_expand_layer = DynamicConvLayer(
                in_channel_list=feature_dim, out_channel_list=final_expand_width, kernel_size=1, act_func='h_swish'
            )
            feature_mix_layer = DynamicConvLayer(
                in_channel_list=final_expand_width, out_channel_list=last_channel, kernel_size=1,
                use_bn=False, act_func='h_swish',
            )
        if len(set(last_channel)) == 1:
            classifier = LinearLayer(max(last_channel), n_classes, dropout_rate=dropout_rate)
        else:
            classifier = DynamicLinearLayer(
                in_features_list=last_channel, out_features=n_classes, bias=True, dropout_rate=dropout_rate
            )
        super(OFAMobileNetV3, self).__init__(first_conv, blocks, final_expand_layer, feature_mix_layer, classifier)

        # set bn param
        self.set_bn_param(momentum=bn_param[0], eps=bn_param[1])

        # runtime_depth
        self.runtime_depth = [len(block_idx) for block_idx in self.block_group_info]

    """ MyNetwork required methods """

    @staticmethod
    def name():
        return 'OFAMobileNetV3'

    def forward(self, x):
        # first conv
        x = self.first_conv(x)
        # first block
        x = self.blocks[0](x)

        # blocks
        for stage_id, block_idx in enumerate(self.block_group_info):
            depth = self.runtime_depth[stage_id]
            active_idx = block_idx[:depth]
            for idx in active_idx:
                x = self.blocks[idx](x)

        x = self.final_expand_layer(x)
        x = x.mean(3, keepdim=True).mean(2, keepdim=True)  # global average pooling
        x = self.feature_mix_layer(x)
        x = torch.squeeze(x)
        x = self.classifier(x)
        return x

    @property
    def module_str(self):
        _str = self.first_conv.module_str + '\n'
        _str += self.blocks[0].module_str + '\n'

        for stage_id, block_idx in enumerate(self.block_group_info):
            depth = self.runtime_depth[stage_id]
            active_idx = block_idx[:depth]
            for idx in active_idx:
                _str += self.blocks[idx].module_str + '\n'

        _str += self.final_expand_layer.module_str + '\n'
        _str += self.feature_mix_layer.module_str + '\n'
        _str += self.classifier.module_str + '\n'
        return _str

    @property
    def config(self):
        return {
            'name': OFAMobileNetV3.__name__,
            'bn': self.get_bn_param(),
            'first_conv': self.first_conv.config,
            'blocks': [
                block.config for block in self.blocks
            ],
            'final_expand_layer': self.final_expand_layer.config,
            'feature_mix_layer': self.feature_mix_layer.config,
            'classifier': self.classifier.config,
        }

    @staticmethod
    def build_from_config(config):
        raise ValueError('do not support this function')

    def load_weights_from_net(self, src_model_dict):
        model_dict = self.state_dict()
        for key in src_model_dict:
            if key in model_dict:
                new_key = key
            elif '.bn.bn.' in key:
                new_key = key.replace('.bn.bn.', '.bn.')
            elif '.conv.conv.weight' in key:
                new_key = key.replace('.conv.conv.weight', '.conv.weight')
            elif '.linear.linear.' in key:
                new_key = key.replace('.linear.linear.', '.linear.')
            ##############################################################################
            elif '.linear.' in key:
                new_key = key.replace('.linear.', '.linear.linear.')
            elif 'bn.' in key:
                new_key = key.replace('bn.', 'bn.bn.')
            elif 'conv.weight' in key:
                new_key = key.replace('conv.weight', 'conv.conv.weight')
            else:
                raise ValueError(key)
            assert new_key in model_dict, '%s' % new_key
            model_dict[new_key] = src_model_dict[key]
        self.load_state_dict(model_dict)

    """ set, sample and get active sub-networks """

    def set_active_subnet(self, wid=None, ks=None, e=None, d=None):
        width_mult_id = int2list(wid, 4 + len(self.block_group_info))
        ks = int2list(ks, len(self.blocks) - 1)
        expand_ratio = int2list(e, len(self.blocks) - 1)
        depth = int2list(d, len(self.block_group_info))

        for block, k, e in zip(self.blocks[1:], ks, expand_ratio):
            if k is not None:
                block.mobile_inverted_conv.active_kernel_size = k
            if e is not None:
                block.mobile_inverted_conv.active_expand_ratio = e

        for i, d in enumerate(depth):
            if d is not None:
                self.runtime_depth[i] = min(len(self.block_group_info[i]), d)

    def set_constraint(self, include_list, constraint_type='depth'):
        if constraint_type == 'depth':
            self.__dict__['_depth_include_list'] = include_list.copy()
        elif constraint_type == 'expand_ratio':
            self.__dict__['_expand_include_list'] = include_list.copy()
        elif constraint_type == 'kernel_size':
            self.__dict__['_ks_include_list'] = include_list.copy()
        elif constraint_type == 'width_mult':
            self.__dict__['_widthMult_include_list'] = include_list.copy()
        else:
            raise NotImplementedError

    def clear_constraint(self):
        self.__dict__['_depth_include_list'] = None
        self.__dict__['_expand_include_list'] = None
        self.__dict__['_ks_include_list'] = None
        self.__dict__['_widthMult_include_list'] = None

    def sample_active_subnet(self):
        ks_candidates = self.ks_list if self.__dict__.get('_ks_include_list', None) is None \
            else self.__dict__['_ks_include_list']
        expand_candidates = self.expand_ratio_list if self.__dict__.get('_expand_include_list', None) is None \
            else self.__dict__['_expand_include_list']
        depth_candidates = self.depth_list if self.__dict__.get('_depth_include_list', None) is None else \
            self.__dict__['_depth_include_list']

        # sample width_mult
        width_mult_setting = None

        # sample kernel size
        ks_setting = []
        if not isinstance(ks_candidates[0], list):
            ks_candidates = [ks_candidates for _ in range(len(self.blocks) - 1)]
        for k_set in ks_candidates:
            k = random.choice(k_set)
            ks_setting.append(k)

        # sample expand ratio
        expand_setting = []
        if not isinstance(expand_candidates[0], list):
            expand_candidates = [expand_candidates for _ in range(len(self.blocks) - 1)]
        for e_set in expand_candidates:
            e = random.choice(e_set)
            expand_setting.append(e)

        # sample depth
        depth_setting = []
        if not isinstance(depth_candidates[0], list):
            depth_candidates = [depth_candidates for _ in range(len(self.block_group_info))]
        for d_set in depth_candidates:
            d = random.choice(d_set)
            depth_setting.append(d)

        self.set_active_subnet(width_mult_setting, ks_setting, expand_setting, depth_setting)

        return {
            'wid': width_mult_setting,
            'ks': ks_setting,
            'e': expand_setting,
            'd': depth_setting,
        }

    def get_active_subnet(self, preserve_weight=True):
        first_conv = copy.deepcopy(self.first_conv)
        blocks = [copy.deepcopy(self.blocks[0])]

        final_expand_layer = copy.deepcopy(self.final_expand_layer)
        feature_mix_layer = copy.deepcopy(self.feature_mix_layer)
        classifier = copy.deepcopy(self.classifier)

        input_channel = blocks[0].mobile_inverted_conv.out_channels
        # blocks
        for stage_id, block_idx in enumerate(self.block_group_info):
            depth = self.runtime_depth[stage_id]
            active_idx = block_idx[:depth]
            stage_blocks = []
            for idx in active_idx:
                stage_blocks.append(MobileInvertedResidualBlock(
                    self.blocks[idx].mobile_inverted_conv.get_active_subnet(input_channel, preserve_weight),
                    copy.deepcopy(self.blocks[idx].shortcut)
                ))
                input_channel = stage_blocks[-1].mobile_inverted_conv.out_channels
            blocks += stage_blocks

        _subnet = MobileNetV3(first_conv, blocks, final_expand_layer, feature_mix_layer, classifier)
        _subnet.set_bn_param(**self.get_bn_param())
        return _subnet

    def get_active_net_config(self):
        # first conv
        first_conv_config = self.first_conv.config
        first_block_config = self.blocks[0].config
        if isinstance(self.first_conv, DynamicConvLayer):
            first_conv_config = self.first_conv.get_active_subnet_config(3)
            first_block_config = {
                'name': MobileInvertedResidualBlock.__name__,
                'mobile_inverted_conv': self.blocks[0].mobile_inverted_conv.get_active_subnet_config(
                    first_conv_config['out_channels']
                ),
                'shortcut': self.blocks[0].shortcut.config if self.blocks[0].shortcut is not None else None,
            }
        final_expand_config = self.final_expand_layer.config
        feature_mix_layer_config = self.feature_mix_layer.config
        if isinstance(self.final_expand_layer, DynamicConvLayer):
            final_expand_config = self.final_expand_layer.get_active_subnet_config(
                self.blocks[-1].mobile_inverted_conv.active_out_channel)
            feature_mix_layer_config = self.feature_mix_layer.get_active_subnet_config(
                final_expand_config['out_channels'])
        classifier_config = self.classifier.config
        if isinstance(self.classifier, DynamicLinearLayer):
            classifier_config = self.classifier.get_active_subnet_config(self.feature_mix_layer.active_out_channel)

        block_config_list = [first_block_config]
        input_channel = first_block_config['mobile_inverted_conv']['out_channels']
        for stage_id, block_idx in enumerate(self.block_group_info):
            depth = self.runtime_depth[stage_id]
            active_idx = block_idx[:depth]
            stage_blocks = []
            for idx in active_idx:
                middle_channel = make_divisible(round(input_channel *
                                                      self.blocks[idx].mobile_inverted_conv.active_expand_ratio), 8)
                stage_blocks.append({
                    'name': MobileInvertedResidualBlock.__name__,
                    'mobile_inverted_conv': {
                        'name': MBInvertedConvLayer.__name__,
                        'in_channels': input_channel,
                        'out_channels': self.blocks[idx].mobile_inverted_conv.active_out_channel,
                        'kernel_size': self.blocks[idx].mobile_inverted_conv.active_kernel_size,
                        'stride': self.blocks[idx].mobile_inverted_conv.stride,
                        'expand_ratio': self.blocks[idx].mobile_inverted_conv.active_expand_ratio,
                        'mid_channels': middle_channel,
                        'act_func': self.blocks[idx].mobile_inverted_conv.act_func,
                        'use_se': self.blocks[idx].mobile_inverted_conv.use_se,
                    },
                    'shortcut': self.blocks[idx].shortcut.config if self.blocks[idx].shortcut is not None else None,
                })
                input_channel = self.blocks[idx].mobile_inverted_conv.active_out_channel
            block_config_list += stage_blocks

        return {
            'name': MobileNetV3.__name__,
            'bn': self.get_bn_param(),
            'first_conv': first_conv_config,
            'blocks': block_config_list,
            'final_expand_layer': final_expand_config,
            'feature_mix_layer': feature_mix_layer_config,
            'classifier': classifier_config,
        }

    """ Width Related Methods """

    def re_organize_middle_weights(self, expand_ratio_stage=0):
        for block in self.blocks[1:]:
            block.mobile_inverted_conv.re_organize_middle_weights(expand_ratio_stage)
