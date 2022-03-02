# Once for All: Train One Network and Specialize it for Efficient Deployment
# Han Cai, Chuang Gan, Tianzhe Wang, Zhekai Zhang, Song Han
# International Conference on Learning Representations (ICLR), 2020.

import math
import torch.nn as nn
import torch.nn.functional as F

from .common_tools import min_divisible_value

__all__ = [
    "MyModule",
    "MyNetwork",
    "init_models",
    "set_bn_param",
    "get_bn_param",
    "replace_bn_with_gn",
    "MyConv2d",
    "replace_conv2d_with_my_conv2d",
]


def set_bn_param(net, momentum, eps, gn_channel_per_group=None, ws_eps=None, **kwargs):
    replace_bn_with_gn(net, gn_channel_per_group)

    for m in net.modules():
        if type(m) in [nn.BatchNorm1d, nn.BatchNorm2d]:
            m.momentum = momentum
            m.eps = eps
        elif isinstance(m, nn.GroupNorm):
            m.eps = eps

    replace_conv2d_with_my_conv2d(net, ws_eps)
    return


def get_bn_param(net):
    ws_eps = None
    for m in net.modules():
        if isinstance(m, MyConv2d):
            ws_eps = m.WS_EPS
            break
    for m in net.modules():
        if isinstance(m, nn.BatchNorm2d) or isinstance(m, nn.BatchNorm1d):
            return {
                "momentum": m.momentum,
                "eps": m.eps,
                "ws_eps": ws_eps,
            }
        elif isinstance(m, nn.GroupNorm):
            return {
                "momentum": None,
                "eps": m.eps,
                "gn_channel_per_group": m.num_channels // m.num_groups,
                "ws_eps": ws_eps,
            }
    return None


def replace_bn_with_gn(model, gn_channel_per_group):
    if gn_channel_per_group is None:
        return

    for m in model.modules():
        to_replace_dict = {}
        for name, sub_m in m.named_children():
            if isinstance(sub_m, nn.BatchNorm2d):
                num_groups = sub_m.num_features // min_divisible_value(
                    sub_m.num_features, gn_channel_per_group
                )
                gn_m = nn.GroupNorm(
                    num_groups=num_groups,
                    num_channels=sub_m.num_features,
                    eps=sub_m.eps,
                    affine=True,
                )

                # load weight
                gn_m.weight.data.copy_(sub_m.weight.data)
                gn_m.bias.data.copy_(sub_m.bias.data)
                # load requires_grad
                gn_m.weight.requires_grad = sub_m.weight.requires_grad
                gn_m.bias.requires_grad = sub_m.bias.requires_grad

                to_replace_dict[name] = gn_m
        m._modules.update(to_replace_dict)


def replace_conv2d_with_my_conv2d(net, ws_eps=None):
    if ws_eps is None:
        return

    for m in net.modules():
        to_update_dict = {}
        for name, sub_module in m.named_children():
            if isinstance(sub_module, nn.Conv2d) and not sub_module.bias:
                # only replace conv2d layers that are followed by normalization layers (i.e., no bias)
                to_update_dict[name] = sub_module
        for name, sub_module in to_update_dict.items():
            m._modules[name] = MyConv2d(
                sub_module.in_channels,
                sub_module.out_channels,
                sub_module.kernel_size,
                sub_module.stride,
                sub_module.padding,
                sub_module.dilation,
                sub_module.groups,
                sub_module.bias,
            )
            # load weight
            m._modules[name].load_state_dict(sub_module.state_dict())
            # load requires_grad
            m._modules[name].weight.requires_grad = sub_module.weight.requires_grad
            if sub_module.bias is not None:
                m._modules[name].bias.requires_grad = sub_module.bias.requires_grad
    # set ws_eps
    for m in net.modules():
        if isinstance(m, MyConv2d):
            m.WS_EPS = ws_eps


def init_models(net, model_init="he_fout"):
    """
    Conv2d,
    BatchNorm2d, BatchNorm1d, GroupNorm
    Linear,
    """
    if isinstance(net, list):
        for sub_net in net:
            init_models(sub_net, model_init)
        return
    for m in net.modules():
        if isinstance(m, nn.Conv2d):
            if model_init == "he_fout":
                n = m.kernel_size[0] * m.kernel_size[1] * m.out_channels
                m.weight.data.normal_(0, math.sqrt(2.0 / n))
            elif model_init == "he_fin":
                n = m.kernel_size[0] * m.kernel_size[1] * m.in_channels
                m.weight.data.normal_(0, math.sqrt(2.0 / n))
            else:
                raise NotImplementedError
            if m.bias is not None:
                m.bias.data.zero_()
        elif type(m) in [nn.BatchNorm2d, nn.BatchNorm1d, nn.GroupNorm]:
            m.weight.data.fill_(1)
            m.bias.data.zero_()
        elif isinstance(m, nn.Linear):
            stdv = 1.0 / math.sqrt(m.weight.size(1))
            m.weight.data.uniform_(-stdv, stdv)
            if m.bias is not None:
                m.bias.data.zero_()


class MyConv2d(nn.Conv2d):
    """
    Conv2d with Weight Standardization
    https://github.com/joe-siyuan-qiao/WeightStandardization
    """

    def __init__(
        self,
        in_channels,
        out_channels,
        kernel_size,
        stride=1,
        padding=0,
        dilation=1,
        groups=1,
        bias=True,
    ):
        super(MyConv2d, self).__init__(
            in_channels,
            out_channels,
            kernel_size,
            stride,
            padding,
            dilation,
            groups,
            bias,
        )
        self.WS_EPS = None

    def weight_standardization(self, weight):
        if self.WS_EPS is not None:
            weight_mean = (
                weight.mean(dim=1, keepdim=True)
                .mean(dim=2, keepdim=True)
                .mean(dim=3, keepdim=True)
            )
            weight = weight - weight_mean
            std = (
                weight.view(weight.size(0), -1).std(dim=1).view(-1, 1, 1, 1)
                + self.WS_EPS
            )
            weight = weight / std.expand_as(weight)
        return weight

    def forward(self, x):
        if self.WS_EPS is None:
            return super(MyConv2d, self).forward(x)
        else:
            return F.conv2d(
                x,
                self.weight_standardization(self.weight),
                self.bias,
                self.stride,
                self.padding,
                self.dilation,
                self.groups,
            )

    def __repr__(self):
        return super(MyConv2d, self).__repr__()[:-1] + ", ws_eps=%s)" % self.WS_EPS


class MyModule(nn.Module):
    def forward(self, x):
        raise NotImplementedError

    @property
    def module_str(self):
        raise NotImplementedError

    @property
    def config(self):
        raise NotImplementedError

    @staticmethod
    def build_from_config(config):
        raise NotImplementedError


class MyNetwork(MyModule):
    CHANNEL_DIVISIBLE = 8

    def forward(self, x):
        raise NotImplementedError

    @property
    def module_str(self):
        raise NotImplementedError

    @property
    def config(self):
        raise NotImplementedError

    @staticmethod
    def build_from_config(config):
        raise NotImplementedError

    def zero_last_gamma(self):
        raise NotImplementedError

    @property
    def grouped_block_index(self):
        raise NotImplementedError

    """ implemented methods """

    def set_bn_param(self, momentum, eps, gn_channel_per_group=None, **kwargs):
        set_bn_param(self, momentum, eps, gn_channel_per_group, **kwargs)

    def get_bn_param(self):
        return get_bn_param(self)

    def get_parameters(self, keys=None, mode="include"):
        if keys is None:
            for name, param in self.named_parameters():
                if param.requires_grad:
                    yield param
        elif mode == "include":
            for name, param in self.named_parameters():
                flag = False
                for key in keys:
                    if key in name:
                        flag = True
                        break
                if flag and param.requires_grad:
                    yield param
        elif mode == "exclude":
            for name, param in self.named_parameters():
                flag = True
                for key in keys:
                    if key in name:
                        flag = False
                        break
                if flag and param.requires_grad:
                    yield param
        else:
            raise ValueError("do not support: %s" % mode)

    def weight_parameters(self):
        return self.get_parameters()
