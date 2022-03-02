import math
import torch
from torch.utils.data.distributed import DistributedSampler

__all__ = ["MyDistributedSampler", "WeightedDistributedSampler"]


class MyDistributedSampler(DistributedSampler):
    """Allow Subset Sampler in Distributed Training"""

    def __init__(
        self, dataset, num_replicas=None, rank=None, shuffle=True, sub_index_list=None
    ):
        super(MyDistributedSampler, self).__init__(dataset, num_replicas, rank, shuffle)
        self.sub_index_list = sub_index_list  # numpy

        self.num_samples = int(
            math.ceil(len(self.sub_index_list) * 1.0 / self.num_replicas)
        )
        self.total_size = self.num_samples * self.num_replicas
        print("Use MyDistributedSampler: %d, %d" % (self.num_samples, self.total_size))

    def __iter__(self):
        # deterministically shuffle based on epoch
        g = torch.Generator()
        g.manual_seed(self.epoch)
        indices = torch.randperm(len(self.sub_index_list), generator=g).tolist()

        # add extra samples to make it evenly divisible
        indices += indices[: (self.total_size - len(indices))]
        indices = self.sub_index_list[indices].tolist()
        assert len(indices) == self.total_size

        # subsample
        indices = indices[self.rank : self.total_size : self.num_replicas]
        assert len(indices) == self.num_samples

        return iter(indices)


class WeightedDistributedSampler(DistributedSampler):
    """Allow Weighted Random Sampling in Distributed Training"""

    def __init__(
        self,
        dataset,
        num_replicas=None,
        rank=None,
        shuffle=True,
        weights=None,
        replacement=True,
    ):
        super(WeightedDistributedSampler, self).__init__(
            dataset, num_replicas, rank, shuffle
        )

        self.weights = (
            torch.as_tensor(weights, dtype=torch.double)
            if weights is not None
            else None
        )
        self.replacement = replacement
        print("Use WeightedDistributedSampler")

    def __iter__(self):
        if self.weights is None:
            return super(WeightedDistributedSampler, self).__iter__()
        else:
            g = torch.Generator()
            g.manual_seed(self.epoch)
            if self.shuffle:
                # original: indices = torch.randperm(len(self.dataset), generator=g).tolist()
                indices = torch.multinomial(
                    self.weights, len(self.dataset), self.replacement, generator=g
                ).tolist()
            else:
                indices = list(range(len(self.dataset)))

            # add extra samples to make it evenly divisible
            indices += indices[: (self.total_size - len(indices))]
            assert len(indices) == self.total_size

            # subsample
            indices = indices[self.rank : self.total_size : self.num_replicas]
            assert len(indices) == self.num_samples

            return iter(indices)
