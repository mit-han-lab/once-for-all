# Once for All: Train One Network and Specialize it for Efficient Deployment
# Han Cai, Chuang Gan, Tianzhe Wang, Zhekai Zhang, Song Han
# International Conference on Learning Representations (ICLR), 2020.

import json
import torch

from ofa.utils import download_url
from ofa.imagenet_classification.networks import get_net_by_name, proxyless_base
from ofa.imagenet_classification.elastic_nn.networks import OFAMobileNetV3, OFAProxylessNASNets, OFAResNets

__all__ = [
	'ofa_specialized', 'ofa_net',
	'proxylessnas_net', 'proxylessnas_mobile', 'proxylessnas_cpu', 'proxylessnas_gpu',
]


def ofa_specialized(net_id, pretrained=True):
	url_base = 'https://hanlab.mit.edu/files/OnceForAll/ofa_specialized/'
	net_config = json.load(open(
		download_url(url_base + net_id + '/net.config', model_dir='.torch/ofa_specialized/%s/' % net_id)
	))
	net = get_net_by_name(net_config['name']).build_from_config(net_config)

	image_size = json.load(open(
		download_url(url_base + net_id + '/run.config', model_dir='.torch/ofa_specialized/%s/' % net_id)
	))['image_size']

	if pretrained:
		init = torch.load(
			download_url(url_base + net_id + '/init', model_dir='.torch/ofa_specialized/%s/' % net_id),
			map_location='cpu'
		)['state_dict']
		net.load_state_dict(init)
	return net, image_size


def ofa_net(net_id, pretrained=True):
	if net_id == 'ofa_proxyless_d234_e346_k357_w1.3':
		net = OFAProxylessNASNets(
			dropout_rate=0, width_mult=1.3, ks_list=[3, 5, 7], expand_ratio_list=[3, 4, 6], depth_list=[2, 3, 4],
		)
	elif net_id == 'ofa_mbv3_d234_e346_k357_w1.0':
		net = OFAMobileNetV3(
			dropout_rate=0, width_mult=1.0, ks_list=[3, 5, 7], expand_ratio_list=[3, 4, 6], depth_list=[2, 3, 4],
		)
	elif net_id == 'ofa_mbv3_d234_e346_k357_w1.2':
		net = OFAMobileNetV3(
			dropout_rate=0, width_mult=1.2, ks_list=[3, 5, 7], expand_ratio_list=[3, 4, 6], depth_list=[2, 3, 4],
		)
	elif net_id == 'ofa_resnet50':
		net = OFAResNets(
			dropout_rate=0, depth_list=[0, 1, 2], expand_ratio_list=[0.2, 0.25, 0.35], width_mult_list=[0.65, 0.8, 1.0]
		)
		net_id = 'ofa_resnet50_d=0+1+2_e=0.2+0.25+0.35_w=0.65+0.8+1.0'
	else:
		raise ValueError('Not supported: %s' % net_id)

	if pretrained:
		url_base = 'https://hanlab.mit.edu/files/OnceForAll/ofa_nets/'
		init = torch.load(
			download_url(url_base + net_id, model_dir='.torch/ofa_nets'),
			map_location='cpu')['state_dict']
		net.load_state_dict(init)
	return net


def proxylessnas_net(net_id, pretrained=True):
	net = proxyless_base(
		net_config='https://hanlab.mit.edu/files/proxylessNAS/%s.config' % net_id,
	)
	if pretrained:
		net.load_state_dict(torch.load(
			download_url('https://hanlab.mit.edu/files/proxylessNAS/%s.pth' % net_id), map_location='cpu'
		)['state_dict'])
	return net


def proxylessnas_mobile(pretrained=True):
	return proxylessnas_net('proxyless_mobile', pretrained)


def proxylessnas_cpu(pretrained=True):
	return proxylessnas_net('proxyless_cpu', pretrained)


def proxylessnas_gpu(pretrained=True):
	return proxylessnas_net('proxyless_gpu', pretrained)
