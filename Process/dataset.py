import os
import numpy as np
import torch
import random
from torch.utils.data import Dataset
from torch_geometric.data import Data
from Process.augmentation import *

max_len = 140


class BiGraphDataset(Dataset):
    def __init__(self, fold_x, treeDic, drop_edge_rate_1=0.3, drop_edge_rate_2=0.4, drop_feature_rate_1=0.1,
                 drop_feature_rate_2=0.0, lower=2, upper=100000, tddroprate=0, budroprate=0,
                 data_path=os.path.join('..', '..', 'data', 'Weibograph')):
        self.fold_x = list(
            filter(lambda id: id in treeDic and len(treeDic[id]) >= lower and len(treeDic[id]) <= upper, fold_x))
        self.treeDic = treeDic
        self.data_path = data_path
        self.tddroprate = tddroprate
        self.budroprate = budroprate
        self.drop_edge_rate_1 = drop_edge_rate_1
        self.drop_edge_rate_2 = drop_edge_rate_2
        self.drop_feature_rate_1 = drop_feature_rate_1
        self.drop_feature_rate_2 = drop_feature_rate_2

    def __len__(self):
        return len(self.fold_x)

    def __getitem__(self, index):
        id = self.fold_x[index]
        data = np.load(os.path.join(self.data_path, id + ".npz"), allow_pickle=True)
        edgeindex = torch.LongTensor(data['edgeindex'])
        seqlen = torch.LongTensor([(data['seqlen'])]).squeeze()

        x_features = torch.tensor([item.detach().numpy() for item in list(data['x'])], dtype=torch.float32)
        # In addition to the original image, three enhancements are made
        # First, node properties are randomly shuffled among all nodes in the graph.
        idx = np.random.permutation(x_features.shape[0])
        aug1_x = x_features[idx, :]

        # Second, adaptive edge dropping; Third, adaptive node attribute masking
        drop_weights = degree_drop_weights(edgeindex)
        edge_index_ = to_undirected(edgeindex)
        node_deg = degree(edge_index_[1])
        feature_weights = feature_drop_weights(x_features, node_c=node_deg)

        def drop_edge(drop_weights, idx: int):
            if idx == 1:
                return drop_edge_weighted(edgeindex, drop_weights, p=self.drop_edge_rate_1, threshold=0.7)
            elif idx == 2:
                return drop_edge_weighted(edgeindex, drop_weights, p=self.drop_edge_rate_2, threshold=0.7)

        aug2_edge_index = drop_edge(drop_weights, 1)
        aug3_edge_index = drop_edge(drop_weights, 2)
        aug2_x = drop_feature_weighted_2(x_features, feature_weights, self.drop_feature_rate_1)
        aug3_x = drop_feature_weighted_2(x_features, feature_weights, self.drop_feature_rate_2)

        if self.tddroprate > 0:
            row = list(edgeindex[0])
            col = list(edgeindex[1])
            length = len(row)
            poslist = random.sample(range(length), int(length * (1 - self.tddroprate)))
            poslist = sorted(poslist)
            row = list(np.array(row)[poslist])
            col = list(np.array(col)[poslist])
            new_edgeindex = [row, col]
        else:
            new_edgeindex = edgeindex

        burow = list(edgeindex[1])
        bucol = list(edgeindex[0])
        if self.budroprate > 0:
            length = len(burow)
            poslist = random.sample(range(length), int(length * (1 - self.budroprate)))
            poslist = sorted(poslist)
            row = list(np.array(burow)[poslist])
            col = list(np.array(bucol)[poslist])
            bunew_edgeindex = [row, col]
        else:
            bunew_edgeindex = [burow, bucol]
        return Data(x=x_features, edge_index=torch.LongTensor(new_edgeindex),
                    BU_edge_index=torch.LongTensor(bunew_edgeindex), y=torch.LongTensor([int(data['y'])]),
                    root=torch.from_numpy(data['root']), rootindex=torch.LongTensor([int(data['rootindex'])]),
                    seqlen=seqlen, aug1_x=aug1_x, aug2_x=aug2_x, aug3_x=aug3_x, aug2_edge_index=aug2_edge_index,
                    aug3_edge_index=aug3_edge_index, ori_edge_index=edgeindex)
