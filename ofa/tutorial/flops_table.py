import time
import copy
import torch
import torch.nn as nn
import numpy as np
from ofa.utils.layers import *

__all__ = ['FLOPsTable']


def rm_bn_from_net(net):
	for m in net.modules():
		if isinstance(m, nn.BatchNorm2d) or isinstance(m, nn.BatchNorm1d):
			m.forward = lambda x: x


class FLOPsTable:
	def __init__(self, pred_type='flops', device='cuda:0', multiplier=1.2, batch_size=64, load_efficiency_table=None):
		assert pred_type in ['flops', 'latency']
		self.multiplier = multiplier
		self.pred_type = pred_type
		self.device = device
		self.batch_size = batch_size
		self.efficiency_dict = {}
		if load_efficiency_table is not None:
			self.efficiency_dict = np.load(load_efficiency_table, allow_pickle=True).item()
		else:
			self.build_lut(batch_size)

	@torch.no_grad()
	def measure_single_layer_latency(self, layer: nn.Module, input_size: tuple, warmup_steps=10, measure_steps=50):
		total_time = 0
		inputs = torch.randn(*input_size, device=self.device)
		layer.eval()
		rm_bn_from_net(layer)
		network = layer.to(self.device)
		torch.cuda.synchronize()
		for i in range(warmup_steps):
			network(inputs)
		torch.cuda.synchronize()

		torch.cuda.synchronize()
		st = time.time()
		for i in range(measure_steps):
			network(inputs)
		torch.cuda.synchronize()
		ed = time.time()
		total_time += ed - st

		latency = total_time / measure_steps * 1000

		return latency

	@torch.no_grad()
	def measure_single_layer_flops(self, layer: nn.Module, input_size: tuple):
		import thop
		inputs = torch.randn(*input_size, device=self.device)
		network = layer.to(self.device)
		layer.eval()
		rm_bn_from_net(layer)
		flops, params = thop.profile(network, (inputs,), verbose=False)
		return flops / 1e6

	def build_lut(self, batch_size=1, resolutions=[160, 176, 192, 208, 224]):
		for resolution in resolutions:
			self.build_single_lut(batch_size, resolution)

		np.save('local_lut.npy', self.efficiency_dict)

	def build_single_lut(self, batch_size=1, base_resolution=224):
		print('Building the %s lookup table (resolution=%d)...' % (self.pred_type, base_resolution))
		# block, input_size, in_channels, out_channels, expand_ratio, kernel_size, stride, act, se
		configurations = [
			(ConvLayer, base_resolution, 3, 16, 3, 2, 'relu'),
			(ResidualBlock, base_resolution // 2, 16, 16, [1], [3, 5, 7], 1, 'relu', False),
			(ResidualBlock, base_resolution // 2, 16, 24, [3, 4, 6], [3, 5, 7], 2, 'relu', False),
			(ResidualBlock, base_resolution // 4, 24, 24, [3, 4, 6], [3, 5, 7], 1, 'relu', False),
			(ResidualBlock, base_resolution // 4, 24, 24, [3, 4, 6], [3, 5, 7], 1, 'relu', False),
			(ResidualBlock, base_resolution // 4, 24, 24, [3, 4, 6], [3, 5, 7], 1, 'relu', False),
			(ResidualBlock, base_resolution // 4, 24, 40, [3, 4, 6], [3, 5, 7], 2, 'relu', True),
			(ResidualBlock, base_resolution // 8, 40, 40, [3, 4, 6], [3, 5, 7], 1, 'relu', True),
			(ResidualBlock, base_resolution // 8, 40, 40, [3, 4, 6], [3, 5, 7], 1, 'relu', True),
			(ResidualBlock, base_resolution // 8, 40, 40, [3, 4, 6], [3, 5, 7], 1, 'relu', True),
			(ResidualBlock, base_resolution // 8, 40, 80, [3, 4, 6], [3, 5, 7], 2, 'h_swish', False),
			(ResidualBlock, base_resolution // 16, 80, 80, [3, 4, 6], [3, 5, 7], 1, 'h_swish', False),
			(ResidualBlock, base_resolution // 16, 80, 80, [3, 4, 6], [3, 5, 7], 1, 'h_swish', False),
			(ResidualBlock, base_resolution // 16, 80, 80, [3, 4, 6], [3, 5, 7], 1, 'h_swish', False),
			(ResidualBlock, base_resolution // 16, 80, 112, [3, 4, 6], [3, 5, 7], 1, 'h_swish', True),
			(ResidualBlock, base_resolution // 16, 112, 112, [3, 4, 6], [3, 5, 7], 1, 'h_swish', True),
			(ResidualBlock, base_resolution // 16, 112, 112, [3, 4, 6], [3, 5, 7], 1, 'h_swish', True),
			(ResidualBlock, base_resolution // 16, 112, 112, [3, 4, 6], [3, 5, 7], 1, 'h_swish', True),
			(ResidualBlock, base_resolution // 16, 112, 160, [3, 4, 6], [3, 5, 7], 2, 'h_swish', True),
			(ResidualBlock, base_resolution // 32, 160, 160, [3, 4, 6], [3, 5, 7], 1, 'h_swish', True),
			(ResidualBlock, base_resolution // 32, 160, 160, [3, 4, 6], [3, 5, 7], 1, 'h_swish', True),
			(ResidualBlock, base_resolution // 32, 160, 160, [3, 4, 6], [3, 5, 7], 1, 'h_swish', True),
			(ConvLayer, base_resolution // 32, 160, 960, 1, 1, 'h_swish'),
			(ConvLayer, 1, 960, 1280, 1, 1, 'h_swish'),
			(LinearLayer, 1, 1280, 1000, 1, 1),
		]

		efficiency_dict = {
			'mobile_inverted_blocks': [],
			'other_blocks': {}
		}

		for layer_idx in range(len(configurations)):
			config = configurations[layer_idx]
			op_type = config[0]
			if op_type == ResidualBlock:
				_, input_size, in_channels, out_channels, expand_list, ks_list, stride, act, se = config
				in_channels = int(round(in_channels * self.multiplier))
				out_channels = int(round(out_channels * self.multiplier))
				template_config = {
					'name': ResidualBlock.__name__,
					'mobile_inverted_conv': {
						'name': MBConvLayer.__name__,
						'in_channels': in_channels,
						'out_channels': out_channels,
						'kernel_size': kernel_size,
						'stride': stride,
						'expand_ratio': 0,
						# 'mid_channels': None,
						'act_func': act,
						'use_se': se,
					},
					'shortcut': {
						'name': IdentityLayer.__name__,
						'in_channels': in_channels,
						'out_channels': out_channels,
					} if (in_channels == out_channels and stride == 1) else None
				}
				sub_dict = {}
				for ks in ks_list:
					for e in expand_list:
						build_config = copy.deepcopy(template_config)
						build_config['mobile_inverted_conv']['expand_ratio'] = e
						build_config['mobile_inverted_conv']['kernel_size'] = ks

						layer = ResidualBlock.build_from_config(build_config)
						input_shape = (batch_size, in_channels, input_size, input_size)

						if self.pred_type == 'flops':
							measure_result = self.measure_single_layer_flops(layer, input_shape) / batch_size
						elif self.pred_type == 'latency':
							measure_result = self.measure_single_layer_latency(layer, input_shape)

						sub_dict[(ks, e)] = measure_result

				efficiency_dict['mobile_inverted_blocks'].append(sub_dict)

			elif op_type == ConvLayer:
				_, input_size, in_channels, out_channels, kernel_size, stride, activation = config
				in_channels = int(round(in_channels * self.multiplier))
				out_channels = int(round(out_channels * self.multiplier))
				build_config = {
					# 'name': ConvLayer.__name__,
					'in_channels': in_channels,
					'out_channels': out_channels,
					'kernel_size': kernel_size,
					'stride': stride,
					'dilation': 1,
					'groups': 1,
					'bias': False,
					'use_bn': True,
					'has_shuffle': False,
					'act_func': activation,
				}
				layer = ConvLayer.build_from_config(build_config)
				input_shape = (batch_size, in_channels, input_size, input_size)

				if self.pred_type == 'flops':
					measure_result = self.measure_single_layer_flops(layer, input_shape) / batch_size
				elif self.pred_type == 'latency':
					measure_result = self.measure_single_layer_latency(layer, input_shape)

				efficiency_dict['other_blocks'][layer_idx] = measure_result

			elif op_type == LinearLayer:
				_, input_size, in_channels, out_channels, kernel_size, stride = config
				in_channels = int(round(in_channels * self.multiplier))
				out_channels = int(round(out_channels * self.multiplier))
				build_config = {
					# 'name': LinearLayer.__name__,
					'in_features': in_channels,
					'out_features': out_channels
				}
				layer = LinearLayer.build_from_config(build_config)
				input_shape = (batch_size, in_channels)

				if self.pred_type == 'flops':
					measure_result = self.measure_single_layer_flops(layer, input_shape) / batch_size
				elif self.pred_type == 'latency':
					measure_result = self.measure_single_layer_latency(layer, input_shape)

				efficiency_dict['other_blocks'][layer_idx] = measure_result

			else:
				raise NotImplementedError

		self.efficiency_dict[base_resolution] = efficiency_dict
		print('Built the %s lookup table (resolution=%d)!' % (self.pred_type, base_resolution))
		return efficiency_dict

	def predict_efficiency(self, sample):
		input_size = sample.get('r', [224])
		input_size = input_size[0]
		assert 'ks' in sample and 'e' in sample and 'd' in sample
		assert len(sample['ks']) == len(sample['e']) and len(sample['ks']) == 20
		assert len(sample['d']) == 5
		total_stats = 0.
		for i in range(20):
			stage = i // 4
			depth_max = sample['d'][stage]
			depth = i % 4 + 1
			if depth > depth_max:
				continue
			ks, e = sample['ks'][i], sample['e'][i]
			total_stats += self.efficiency_dict[input_size]['mobile_inverted_blocks'][i + 1][(ks, e)]

		for key in self.efficiency_dict[input_size]['other_blocks']:
			total_stats += self.efficiency_dict[input_size]['other_blocks'][key]

		total_stats += self.efficiency_dict[input_size]['mobile_inverted_blocks'][0][(3, 1)]
		return total_stats
