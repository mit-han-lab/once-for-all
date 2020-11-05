# Once for All: Train One Network and Specialize it for Efficient Deployment
# Han Cai, Chuang Gan, Tianzhe Wang, Zhekai Zhang, Song Han
# International Conference on Learning Representations (ICLR), 2020.

import numpy as np
import torch

__all__ = ['DataProvider']


class DataProvider:
	SUB_SEED = 937162211  # random seed for sampling subset
	VALID_SEED = 2147483647  # random seed for the validation set

	@staticmethod
	def name():
		""" Return name of the dataset """
		raise NotImplementedError

	@property
	def data_shape(self):
		""" Return shape as python list of one data entry """
		raise NotImplementedError

	@property
	def n_classes(self):
		""" Return `int` of num classes """
		raise NotImplementedError

	@property
	def save_path(self):
		""" local path to save the data """
		raise NotImplementedError

	@property
	def data_url(self):
		""" link to download the data """
		raise NotImplementedError

	@staticmethod
	def random_sample_valid_set(train_size, valid_size):
		assert train_size > valid_size

		g = torch.Generator()
		g.manual_seed(DataProvider.VALID_SEED)  # set random seed before sampling validation set
		rand_indexes = torch.randperm(train_size, generator=g).tolist()

		valid_indexes = rand_indexes[:valid_size]
		train_indexes = rand_indexes[valid_size:]
		return train_indexes, valid_indexes

	@staticmethod
	def labels_to_one_hot(n_classes, labels):
		new_labels = np.zeros((labels.shape[0], n_classes), dtype=np.float32)
		new_labels[range(labels.shape[0]), labels] = np.ones(labels.shape)
		return new_labels
