import os
from Process.dataset import BiGraphDataset
import csv
import itertools
import torch as th
import numpy as np
import torch.nn.functional as F
from torch_geometric.nn import GCNConv, SGConv, SAGEConv, GATConv, GraphConv, GINConv


################################### load tree#####################################
def loadTree(data_path, dataname):
    treePath = os.path.join(data_path, 'Weibo/Weibo_covid19_data_all.txt')
    print("reading Weibo_covid19 tree")
    treeDic = {}
    for line in open(treePath):
        line = line.strip('\n')
        line = line.rstrip()
        eid, indexP, indexC = line.split('\t')[0], line.split('\t')[1], int(line.split('\t')[2])
        time_delay, text = float(line.split('\t')[3]), str(line.split('\t')[4])
        if not treeDic.__contains__(eid):
            treeDic[eid] = {}
        treeDic[eid][indexC] = {'parent': indexP, 'time_delay': time_delay, 'text': text}
    print('weibo tree no:', len(treeDic))

    tree_path_twitter = os.path.join(data_path, 'Twitter/Twitter_data_all.txt')
    print("reading twitter tree")
    for line in open(tree_path_twitter):
        line = line.strip('\n')
        line = line.rstrip()
        eid, indexP, indexC = line.split('\t')[0], line.split('\t')[1], int(line.split('\t')[2])
        time_delay, text = float(line.split('\t')[3]), str(line.split('\t')[4])
        if not treeDic.__contains__(eid):
            treeDic[eid] = {}
        treeDic[eid][indexC] = {'parent': indexP, 'time_delay': time_delay, 'text': text}
    print('total tree no:', len(treeDic))

    return treeDic


################################# load data ###################################
def loadBiData(dataname, treeDic, fold_x_train, fold_x_test, TDdroprate, BUdroprate, dataPath, drop_edge_rate_1,
               drop_edge_rate_2, drop_feature_rate_1, drop_feature_rate_2):
    data_path = os.path.join(dataPath, dataname + 'graph')
    print("loading train set", )
    traindata_list = BiGraphDataset(fold_x_train, treeDic, drop_edge_rate_1, drop_edge_rate_2, drop_feature_rate_1,
                                    drop_feature_rate_2, tddroprate=TDdroprate, budroprate=BUdroprate,
                                    data_path=data_path)
    print("train no:", len(traindata_list))
    if len(fold_x_test) > 0:
        print("loading test set", )
        testdata_list = BiGraphDataset(fold_x_test, treeDic, drop_edge_rate_1, drop_edge_rate_2, drop_feature_rate_1,
                                       drop_feature_rate_2, data_path=data_path)
        print("test no:", len(testdata_list))
        return traindata_list, testdata_list
    else:
        return traindata_list


def save_all_folds_to_csv(data, filename):
    max_len = max(len(fold) for fold in data)
    equal_len_folds = []
    for fold in data:
        equal_len_fold = fold + [''] * (max_len - len(fold))  # 在末尾添加空值
        equal_len_folds.append(equal_len_fold)
    with open(filename, 'w', newline='') as f:
        writer = csv.writer(f)
        writer.writerow(
            ['fold0_train', 'fold0_test', 'fold1_train', 'fold1_test', 'fold2_train', 'fold2_test', 'fold3_train',
             'fold3_test', 'fold4_train', 'fold4_test', 'twitter_train'])
        rows = itertools.zip_longest(*equal_len_folds, fillvalue='')
        for row in rows:
            writer.writerow(row)


def sparse_mx_to_torch_sparse_tensor(sparse_mx):
    """Convert a scipy sparse matrix to a torch sparse tensor."""
    sparse_mx = sparse_mx.tocoo().astype(np.float32)
    indices = th.from_numpy(
        np.vstack((sparse_mx.row, sparse_mx.col)).astype(np.int64))
    values = th.from_numpy(sparse_mx.data)
    shape = th.Size(sparse_mx.shape)
    return th.sparse.FloatTensor(indices, values, shape)


def get_base_model(name: str):
    def gat_wrapper(in_channels, out_channels):
        return GATConv(
            in_channels=in_channels,
            out_channels=out_channels // 4,
            heads=4
        )

    def gin_wrapper(in_channels, out_channels):
        mlp = th.nn.Sequential(
            th.nn.Linear(in_channels, 2 * out_channels),
            th.nn.ELU(),
            th.nn.Linear(2 * out_channels, out_channels)
        )
        return GINConv(mlp)

    base_models = {
        'GCNConv': GCNConv,
        'SGConv': SGConv,
        'SAGEConv': SAGEConv,
        'GATConv': gat_wrapper,
        'GraphConv': GraphConv,
        'GINConv': gin_wrapper
    }

    return base_models[name]


def get_activation(name: str):
    activations = {
        'relu': F.relu,
        'hardtanh': F.hardtanh,
        'elu': F.elu,
        'leakyrelu': F.leaky_relu,
        'prelu': th.nn.PReLU(),
        'rrelu': F.rrelu
    }

    return activations[name]
