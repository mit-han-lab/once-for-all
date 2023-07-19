import torch.nn as nn
import torch

import copy

from ofa.utils import download_url


# Helper for constructing the one-hot vectors.
def construct_maps(keys):
    d = dict()
    keys = list(set(keys))
    for k in keys:
        if k not in d:
            d[k] = len(list(d.keys()))
    return d


ks_map = construct_maps(keys=(3, 5, 7))
ex_map = construct_maps(keys=(3, 4, 6))
dp_map = construct_maps(keys=(2, 3, 4))


class AccuracyPredictor:
    def __init__(self, pretrained=True, device="cuda:0"):
        self.device = device

        self.model = nn.Sequential(
            nn.Linear(128, 400),
            nn.ReLU(),
            nn.Linear(400, 400),
            nn.ReLU(),
            nn.Linear(400, 400),
            nn.ReLU(),
            nn.Linear(400, 1),
        )
        if pretrained:
            # load pretrained model
            fname = download_url(
                "https://raw.githubusercontent.com/han-cai/files/master/ofa/acc_predictor.pth"
            )
            self.model.load_state_dict(
                torch.load(fname, map_location=torch.device("cpu"))
            )
        self.model = self.model.to(self.device)

    # TODO: merge it with serialization utils.
    @torch.no_grad()
    def predict_accuracy(self, population):
        all_feats = []
        for sample in population:
            ks_list = copy.deepcopy(sample["ks"])
            ex_list = copy.deepcopy(sample["e"])
            d_list = copy.deepcopy(sample["d"])
            r = copy.deepcopy(sample["r"])[0]
            feats = (
                AccuracyPredictor.spec2feats(ks_list, ex_list, d_list, r)
                .reshape(1, -1)
                .to(self.device)
            )
            all_feats.append(feats)
        all_feats = torch.cat(all_feats, 0)
        pred = self.model(all_feats).cpu()
        return pred

    @staticmethod
    def spec2feats(ks_list, ex_list, d_list, r):
        # This function converts a network config to a feature vector (128-D).
        start = 0
        end = 4
        for d in d_list:
            for j in range(start + d, end):
                ks_list[j] = 0
                ex_list[j] = 0
            start += 4
            end += 4

        # convert to onehot
        ks_onehot = [0 for _ in range(60)]
        ex_onehot = [0 for _ in range(60)]
        r_onehot = [0 for _ in range(8)]

        for i in range(20):
            start = i * 3
            if ks_list[i] != 0:
                ks_onehot[start + ks_map[ks_list[i]]] = 1
            if ex_list[i] != 0:
                ex_onehot[start + ex_map[ex_list[i]]] = 1

        r_onehot[(r - 112) // 16] = 1
        return torch.Tensor(ks_onehot + ex_onehot + r_onehot)
