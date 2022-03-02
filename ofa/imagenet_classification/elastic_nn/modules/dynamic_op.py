# Once for All: Train One Network and Specialize it for Efficient Deployment
# Han Cai, Chuang Gan, Tianzhe Wang, Zhekai Zhang, Song Han
# International Conference on Learning Representations (ICLR), 2020.

import torch.nn.functional as F
import torch.nn as nn
import torch
from torch.nn.parameter import Parameter

from ofa.utils import (
    get_same_padding,
    sub_filter_start_end,
    make_divisible,
    SEModule,
    MyNetwork,
    MyConv2d,
)

__all__ = [
    "DynamicSeparableConv2d",
    "DynamicConv2d",
    "DynamicGroupConv2d",
    "DynamicBatchNorm2d",
    "DynamicGroupNorm",
    "DynamicSE",
    "DynamicLinear",
]


class DynamicSeparableConv2d(nn.Module):
    KERNEL_TRANSFORM_MODE = 1  # None or 1

    def __init__(self, max_in_channels, kernel_size_list, stride=1, dilation=1):
        super(DynamicSeparableConv2d, self).__init__()

        self.max_in_channels = max_in_channels
        self.kernel_size_list = kernel_size_list
        self.stride = stride
        self.dilation = dilation

        self.conv = nn.Conv2d(
            self.max_in_channels,
            self.max_in_channels,
            max(self.kernel_size_list),
            self.stride,
            groups=self.max_in_channels,
            bias=False,
        )

        self._ks_set = list(set(self.kernel_size_list))
        self._ks_set.sort()  # e.g., [3, 5, 7]
        if self.KERNEL_TRANSFORM_MODE is not None:
            # register scaling parameters
            # 7to5_matrix, 5to3_matrix
            scale_params = {}
            for i in range(len(self._ks_set) - 1):
                ks_small = self._ks_set[i]
                ks_larger = self._ks_set[i + 1]
                param_name = "%dto%d" % (ks_larger, ks_small)
                # noinspection PyArgumentList
                scale_params["%s_matrix" % param_name] = Parameter(
                    torch.eye(ks_small ** 2)
                )
            for name, param in scale_params.items():
                self.register_parameter(name, param)

        self.active_kernel_size = max(self.kernel_size_list)

    def get_active_filter(self, in_channel, kernel_size):
        out_channel = in_channel
        max_kernel_size = max(self.kernel_size_list)

        start, end = sub_filter_start_end(max_kernel_size, kernel_size)
        filters = self.conv.weight[:out_channel, :in_channel, start:end, start:end]
        if self.KERNEL_TRANSFORM_MODE is not None and kernel_size < max_kernel_size:
            start_filter = self.conv.weight[
                :out_channel, :in_channel, :, :
            ]  # start with max kernel
            for i in range(len(self._ks_set) - 1, 0, -1):
                src_ks = self._ks_set[i]
                if src_ks <= kernel_size:
                    break
                target_ks = self._ks_set[i - 1]
                start, end = sub_filter_start_end(src_ks, target_ks)
                _input_filter = start_filter[:, :, start:end, start:end]
                _input_filter = _input_filter.contiguous()
                _input_filter = _input_filter.view(
                    _input_filter.size(0), _input_filter.size(1), -1
                )
                _input_filter = _input_filter.view(-1, _input_filter.size(2))
                _input_filter = F.linear(
                    _input_filter,
                    self.__getattr__("%dto%d_matrix" % (src_ks, target_ks)),
                )
                _input_filter = _input_filter.view(
                    filters.size(0), filters.size(1), target_ks ** 2
                )
                _input_filter = _input_filter.view(
                    filters.size(0), filters.size(1), target_ks, target_ks
                )
                start_filter = _input_filter
            filters = start_filter
        return filters

    def forward(self, x, kernel_size=None):
        if kernel_size is None:
            kernel_size = self.active_kernel_size
        in_channel = x.size(1)

        filters = self.get_active_filter(in_channel, kernel_size).contiguous()

        padding = get_same_padding(kernel_size)
        filters = (
            self.conv.weight_standardization(filters)
            if isinstance(self.conv, MyConv2d)
            else filters
        )
        y = F.conv2d(x, filters, None, self.stride, padding, self.dilation, in_channel)
        return y


class DynamicConv2d(nn.Module):
    def __init__(
        self, max_in_channels, max_out_channels, kernel_size=1, stride=1, dilation=1
    ):
        super(DynamicConv2d, self).__init__()

        self.max_in_channels = max_in_channels
        self.max_out_channels = max_out_channels
        self.kernel_size = kernel_size
        self.stride = stride
        self.dilation = dilation

        self.conv = nn.Conv2d(
            self.max_in_channels,
            self.max_out_channels,
            self.kernel_size,
            stride=self.stride,
            bias=False,
        )

        self.active_out_channel = self.max_out_channels

    def get_active_filter(self, out_channel, in_channel):
        return self.conv.weight[:out_channel, :in_channel, :, :]

    def forward(self, x, out_channel=None):
        if out_channel is None:
            out_channel = self.active_out_channel
        in_channel = x.size(1)
        filters = self.get_active_filter(out_channel, in_channel).contiguous()

        padding = get_same_padding(self.kernel_size)
        filters = (
            self.conv.weight_standardization(filters)
            if isinstance(self.conv, MyConv2d)
            else filters
        )
        y = F.conv2d(x, filters, None, self.stride, padding, self.dilation, 1)
        return y


class DynamicGroupConv2d(nn.Module):
    def __init__(
        self,
        in_channels,
        out_channels,
        kernel_size_list,
        groups_list,
        stride=1,
        dilation=1,
    ):
        super(DynamicGroupConv2d, self).__init__()

        self.in_channels = in_channels
        self.out_channels = out_channels
        self.kernel_size_list = kernel_size_list
        self.groups_list = groups_list
        self.stride = stride
        self.dilation = dilation

        self.conv = nn.Conv2d(
            self.in_channels,
            self.out_channels,
            max(self.kernel_size_list),
            self.stride,
            groups=min(self.groups_list),
            bias=False,
        )

        self.active_kernel_size = max(self.kernel_size_list)
        self.active_groups = min(self.groups_list)

    def get_active_filter(self, kernel_size, groups):
        start, end = sub_filter_start_end(max(self.kernel_size_list), kernel_size)
        filters = self.conv.weight[:, :, start:end, start:end]

        sub_filters = torch.chunk(filters, groups, dim=0)
        sub_in_channels = self.in_channels // groups
        sub_ratio = filters.size(1) // sub_in_channels

        filter_crops = []
        for i, sub_filter in enumerate(sub_filters):
            part_id = i % sub_ratio
            start = part_id * sub_in_channels
            filter_crops.append(sub_filter[:, start : start + sub_in_channels, :, :])
        filters = torch.cat(filter_crops, dim=0)
        return filters

    def forward(self, x, kernel_size=None, groups=None):
        if kernel_size is None:
            kernel_size = self.active_kernel_size
        if groups is None:
            groups = self.active_groups

        filters = self.get_active_filter(kernel_size, groups).contiguous()
        padding = get_same_padding(kernel_size)
        filters = (
            self.conv.weight_standardization(filters)
            if isinstance(self.conv, MyConv2d)
            else filters
        )
        y = F.conv2d(
            x,
            filters,
            None,
            self.stride,
            padding,
            self.dilation,
            groups,
        )
        return y


class DynamicBatchNorm2d(nn.Module):
    SET_RUNNING_STATISTICS = False

    def __init__(self, max_feature_dim):
        super(DynamicBatchNorm2d, self).__init__()

        self.max_feature_dim = max_feature_dim
        self.bn = nn.BatchNorm2d(self.max_feature_dim)

    @staticmethod
    def bn_forward(x, bn: nn.BatchNorm2d, feature_dim):
        if bn.num_features == feature_dim or DynamicBatchNorm2d.SET_RUNNING_STATISTICS:
            return bn(x)
        else:
            exponential_average_factor = 0.0

            if bn.training and bn.track_running_stats:
                if bn.num_batches_tracked is not None:
                    bn.num_batches_tracked += 1
                    if bn.momentum is None:  # use cumulative moving average
                        exponential_average_factor = 1.0 / float(bn.num_batches_tracked)
                    else:  # use exponential moving average
                        exponential_average_factor = bn.momentum
            return F.batch_norm(
                x,
                bn.running_mean[:feature_dim],
                bn.running_var[:feature_dim],
                bn.weight[:feature_dim],
                bn.bias[:feature_dim],
                bn.training or not bn.track_running_stats,
                exponential_average_factor,
                bn.eps,
            )

    def forward(self, x):
        feature_dim = x.size(1)
        y = self.bn_forward(x, self.bn, feature_dim)
        return y


class DynamicGroupNorm(nn.GroupNorm):
    def __init__(
        self, num_groups, num_channels, eps=1e-5, affine=True, channel_per_group=None
    ):
        super(DynamicGroupNorm, self).__init__(num_groups, num_channels, eps, affine)
        self.channel_per_group = channel_per_group

    def forward(self, x):
        n_channels = x.size(1)
        n_groups = n_channels // self.channel_per_group
        return F.group_norm(
            x, n_groups, self.weight[:n_channels], self.bias[:n_channels], self.eps
        )

    @property
    def bn(self):
        return self


class DynamicSE(SEModule):
    def __init__(self, max_channel):
        super(DynamicSE, self).__init__(max_channel)

    def get_active_reduce_weight(self, num_mid, in_channel, groups=None):
        if groups is None or groups == 1:
            return self.fc.reduce.weight[:num_mid, :in_channel, :, :]
        else:
            assert in_channel % groups == 0
            sub_in_channels = in_channel // groups
            sub_filters = torch.chunk(
                self.fc.reduce.weight[:num_mid, :, :, :], groups, dim=1
            )
            return torch.cat(
                [sub_filter[:, :sub_in_channels, :, :] for sub_filter in sub_filters],
                dim=1,
            )

    def get_active_reduce_bias(self, num_mid):
        return (
            self.fc.reduce.bias[:num_mid] if self.fc.reduce.bias is not None else None
        )

    def get_active_expand_weight(self, num_mid, in_channel, groups=None):
        if groups is None or groups == 1:
            return self.fc.expand.weight[:in_channel, :num_mid, :, :]
        else:
            assert in_channel % groups == 0
            sub_in_channels = in_channel // groups
            sub_filters = torch.chunk(
                self.fc.expand.weight[:, :num_mid, :, :], groups, dim=0
            )
            return torch.cat(
                [sub_filter[:sub_in_channels, :, :, :] for sub_filter in sub_filters],
                dim=0,
            )

    def get_active_expand_bias(self, in_channel, groups=None):
        if groups is None or groups == 1:
            return (
                self.fc.expand.bias[:in_channel]
                if self.fc.expand.bias is not None
                else None
            )
        else:
            assert in_channel % groups == 0
            sub_in_channels = in_channel // groups
            sub_bias_list = torch.chunk(self.fc.expand.bias, groups, dim=0)
            return torch.cat(
                [sub_bias[:sub_in_channels] for sub_bias in sub_bias_list], dim=0
            )

    def forward(self, x, groups=None):
        in_channel = x.size(1)
        num_mid = make_divisible(
            in_channel // self.reduction, divisor=MyNetwork.CHANNEL_DIVISIBLE
        )

        y = x.mean(3, keepdim=True).mean(2, keepdim=True)
        # reduce
        reduce_filter = self.get_active_reduce_weight(
            num_mid, in_channel, groups=groups
        ).contiguous()
        reduce_bias = self.get_active_reduce_bias(num_mid)
        y = F.conv2d(y, reduce_filter, reduce_bias, 1, 0, 1, 1)
        # relu
        y = self.fc.relu(y)
        # expand
        expand_filter = self.get_active_expand_weight(
            num_mid, in_channel, groups=groups
        ).contiguous()
        expand_bias = self.get_active_expand_bias(in_channel, groups=groups)
        y = F.conv2d(y, expand_filter, expand_bias, 1, 0, 1, 1)
        # hard sigmoid
        y = self.fc.h_sigmoid(y)

        return x * y


class DynamicLinear(nn.Module):
    def __init__(self, max_in_features, max_out_features, bias=True):
        super(DynamicLinear, self).__init__()

        self.max_in_features = max_in_features
        self.max_out_features = max_out_features
        self.bias = bias

        self.linear = nn.Linear(self.max_in_features, self.max_out_features, self.bias)

        self.active_out_features = self.max_out_features

    def get_active_weight(self, out_features, in_features):
        return self.linear.weight[:out_features, :in_features]

    def get_active_bias(self, out_features):
        return self.linear.bias[:out_features] if self.bias else None

    def forward(self, x, out_features=None):
        if out_features is None:
            out_features = self.active_out_features

        in_features = x.size(1)
        weight = self.get_active_weight(out_features, in_features).contiguous()
        bias = self.get_active_bias(out_features)
        y = F.linear(x, weight, bias)
        return y
