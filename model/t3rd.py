import sys, os

sys.path.append(os.getcwd())
from Process.process import *
import torch as th
import numpy as np
import argparse
import gc
import statistics as st
from tools.earlystopping2class import EarlyStopping
from torch_geometric.data import DataLoader
from tqdm import tqdm
from Process.rand5fold import *
from tools.evaluate import *
from Process.distribution_Alignment import offline
from module import Net

cwd = os.getcwd()
default_path = os.path.join(cwd, 'data')

parser = argparse.ArgumentParser(description='T3RD')
parser.add_argument('--trainset', type=str, default='Twitter', help='dataset name')
parser.add_argument('--testset', type=str, default='Weibo_covid19', help='dataset name')
parser.add_argument('--iterations', type=int, default=1, help='iterations')
parser.add_argument('--lr', type=float, default=0.0001, help='learning rate')
parser.add_argument('--weight_decay', type=float, default=1e-4, help='weight decay')
parser.add_argument('--patience', type=int, default=30, help='patience')
parser.add_argument('--n_epochs', type=int, default=200, help='n_epochs')
parser.add_argument('--batchsize', type=int, default=32, help='batchsize')
parser.add_argument('--TDdroprate', type=float, default=0.2, help='TDdroprate')
parser.add_argument('--BUdroprate', type=float, default=0.2, help='BUdroprate')
parser.add_argument('--model', type=str, default='GCN', help='model')
parser.add_argument('--data_path', type=str, default=default_path, help='datatxt_path')
parser.add_argument('--alpha', type=float, default=0.07,
                    help='Training loss = Main task loss + alpha * Auxiliary task loss')
parser.add_argument('--beta', type=float, default=0.6,
                    help='Auxiliary task loss = Global contrastive loss + beta * Local contrastive loss')
parser.add_argument('--gamma', type=float, default=0.7,
                    help='The test-time training loss = Auxiliary task loss + gamma * Adaptive constraint loss')
parser.add_argument('--drop_edge_rate_1', type=float, default=0.0, help='drop_edge_rate_1')
parser.add_argument('--drop_edge_rate_2', type=float, default=0.4, help='drop_edge_rate_2')
parser.add_argument('--drop_feature_rate_1', type=float, default=0.2, help='drop_feature_rate_1')
parser.add_argument('--drop_feature_rate_2', type=float, default=0.0, help='drop_feature_rate_2')
parser.add_argument('--tau', type=float, default=0.1, help='temperature')
parser.add_argument('--fold', type=int, default=5, help='fold')
parser.add_argument('--alp', type=float, default=0.1, help='alp')


def train_GCN(treeDic, x_test, x_train, twitter_train, TDdroprate, BUdroprate, lr, weight_decay, patience, n_epochs,
              batchsize, dataname, iter, data_path, fold_i, alpha, beta, gamma, drop_edge_rate_1, drop_edge_rate_2,
              drop_feature_rate_1, drop_feature_rate_2, tau, alp):
    model = Net(768, 512, 128, tau, 'prelu').to(device)  # 768,512,128

    optimizer = th.optim.AdamW([
        {'params': model.parameters()}
    ], lr=lr, weight_decay=weight_decay)

    test_time_optimizer = th.optim.AdamW([
        {'params': model.TDrumorGCN.parameters()},
        {'params': model.BUrumorGCN.parameters()},
        {'params': model.gcl.parameters()},
        {'params': model.lcl.parameters()}
    ], lr=lr, weight_decay=weight_decay)

    train_losses = []
    val_losses = []
    train_accs = []
    val_accs = []
    early_stopping = EarlyStopping(patience=patience, verbose=True)
    for epoch in range(n_epochs):
        traindata_list, testdata_list = loadBiData(dataname='Weibo', treeDic=treeDic, fold_x_train=x_train,
                                                   fold_x_test=x_test, TDdroprate=TDdroprate, BUdroprate=BUdroprate,
                                                   dataPath=data_path, drop_edge_rate_1=drop_edge_rate_1,
                                                   drop_edge_rate_2=drop_edge_rate_2,
                                                   drop_feature_rate_1=drop_feature_rate_1,
                                                   drop_feature_rate_2=drop_feature_rate_2)
        twitterdata_list = loadBiData(dataname=dataname, treeDic=treeDic, fold_x_train=twitter_train, fold_x_test=[],
                                      TDdroprate=TDdroprate, BUdroprate=BUdroprate, dataPath=data_path,
                                      drop_edge_rate_1=drop_edge_rate_1, drop_edge_rate_2=drop_edge_rate_2,
                                      drop_feature_rate_1=drop_feature_rate_1, drop_feature_rate_2=drop_feature_rate_2)
        train_loader = DataLoader(traindata_list, batch_size=batchsize, shuffle=True, num_workers=5)
        test_loader = DataLoader(testdata_list, batch_size=batchsize, shuffle=True, num_workers=5)
        twitter_loader = DataLoader(twitterdata_list, batch_size=batchsize, shuffle=True, num_workers=5)
        avg_loss = []
        avg_acc = []
        batch_idx = 0
        tqdm_train_loader = tqdm(train_loader)
        flag = ''
        model.train()
        for Batch_data in tqdm_train_loader:
            Batch_data.to(device)
            for Batch_twitter in twitter_loader:
                Batch_twitter.to(device)
                flag = 'training'
                loss, out_labels = model(flag, alpha, beta, gamma, Batch_data, Batch_twitter, alp=alp)

                optimizer.zero_grad()
                loss.backward()
                th.nn.utils.clip_grad_norm_(parameters=model.parameters(), max_norm=5, norm_type=2)
                avg_loss.append(loss.item())
                optimizer.step()
                _, pred = out_labels.max(dim=-1)
                correct = pred.eq(Batch_data.y).sum().item()
                train_acc = correct / len(Batch_data.y)
                avg_acc.append(train_acc)
                postfix = "Iter {:03d} | Epoch {:05d} | Batch{:02d} | Train_Loss {:.4f}| Train_Accuracy {:.4f}".format(
                    iter, epoch, batch_idx,
                    loss.item(),
                    train_acc)
                tqdm_train_loader.set_postfix_str(postfix)
                batch_idx = batch_idx + 1

        train_losses.append(np.mean(avg_loss))
        train_accs.append(np.mean(avg_acc))

        feat_cov_TD, statistics_coral_mean_TD, feat_mean_TD, statistics_mmd_mean_TD = offline(twitter_loader,
                                                                                              model.TDrumorGCN, device)

        temp_test_time_losses = []
        temp_val_losses = []
        temp_val_accs = []

        temp_val_pred = th.tensor([]).to(device)
        temp_val_y = th.tensor([]).to(device)
        # model.eval()
        tqdm_test_loader = tqdm(test_loader)
        for Batch_data in tqdm_test_loader:
            Batch_data.to(device)
            try:
                flag = 'test-time'
                model.train()
                loss = model(flag, alpha, beta, gamma, Batch_data, feat_cov_TD=feat_cov_TD, feat_mean_TD=feat_mean_TD)
                test_time_optimizer.zero_grad()
                loss.backward()
                th.nn.utils.clip_grad_norm_(parameters=model.parameters(), max_norm=5, norm_type=2)
                temp_test_time_losses.append(loss.item())
                test_time_optimizer.step()
            except:
                print('WARNING: out of memory')
                gc.collect()
                th.cuda.empty_cache()

            flag = 'test'
            model.eval()
            val_out = model(flag, alpha, beta, gamma, Batch_data)
            val_loss = F.nll_loss(val_out, Batch_data.y)
            temp_val_losses.append(val_loss.item())
            _, val_pred = val_out.max(dim=1)
            correct = val_pred.eq(Batch_data.y).sum().item()
            val_acc = correct / len(Batch_data.y)
            temp_val_accs.append(val_acc)
            temp_val_pred = th.cat((temp_val_pred, val_pred), 0)
            temp_val_y = th.cat((temp_val_y, Batch_data.y), 0)

        Acc_all, Acc1, Prec1, Recll1, F1, Acc2, Prec2, Recll2, F2 = evaluationclass(temp_val_pred, temp_val_y)
        val_losses.append(np.mean(temp_val_losses))
        val_accs.append(np.mean(temp_val_accs))
        print("Epoch {:05d} | Val_Loss {:.4f}| Val_Accuracy {:.4f}".format(epoch, np.mean(temp_val_losses),
                                                                           np.mean(temp_val_accs)))
        res = ['acc:{:.4f}'.format(Acc_all),
               'C1:{:.4f},{:.4f},{:.4f},{:.4f}'.format(Acc1, Prec1, Recll1, F1),
               'C2:{:.4f},{:.4f},{:.4f},{:.4f}'.format(Acc2, Prec2, Recll2, F2)]
        print('results:', res)
        early_stopping(np.mean(temp_val_losses), Acc_all, Acc1, Acc2, Prec1, Prec2, Recll1, Recll2, F1, F2, model,
                       'T3RD', dataname, fold_i)
        accs = Acc_all
        acc1 = Acc1
        acc2 = Acc2
        pre1 = Prec1
        pre2 = Prec2
        rec1 = Recll1
        rec2 = Recll2
        F1 = F1
        F2 = F2
        if early_stopping.early_stop:
            print("Early stopping")
            accs = early_stopping.accs
            acc1 = early_stopping.acc1
            acc2 = early_stopping.acc2
            pre1 = early_stopping.pre1
            pre2 = early_stopping.pre2
            rec1 = early_stopping.rec1
            rec2 = early_stopping.rec2
            F1 = early_stopping.F1
            F2 = early_stopping.F2
            break
    return train_losses, val_losses, train_accs, val_accs, accs, acc1, pre1, rec1, F1, acc2, pre2, rec2, F2


args = parser.parse_args()
lr = args.lr
weight_decay = args.weight_decay
patience = args.patience
n_epochs = args.n_epochs
batchsize = args.batchsize
TDdroprate = args.TDdroprate
BUdroprate = args.BUdroprate
iterations = args.iterations
model = args.model
trainsetname = args.trainset
data_path = args.data_path
alpha = args.alpha
beta = args.beta
gamma = args.gamma
drop_edge_rate_1 = args.drop_edge_rate_1
drop_edge_rate_2 = args.drop_edge_rate_2
drop_feature_rate_1 = args.drop_feature_rate_1
drop_feature_rate_2 = args.drop_feature_rate_2
tau = args.tau
fold = args.fold
alp = args.alp
device = th.device('cuda:0' if th.cuda.is_available() else 'cpu')
test_accs = []
ACC1, ACC2, PRE1, PRE2, REC1, REC2, F1, F2 = [], [], [], [], [], [], [], []

for iter in range(iterations):
    datasets = loadkfoldData(data_path, trainsetname, fold)
    # save all folds data to csv
    folds_data = [datasets['fold0_train'], datasets['fold0_test'], datasets['fold1_train'], datasets['fold1_test'],
                  datasets['fold2_train'], datasets['fold2_test'], datasets['fold3_train'], datasets['fold3_test'],
                  datasets['fold4_train'], datasets['fold4_test'], datasets['high_resource']]
    folds_data_path = os.path.join(data_path, 'folds_data.csv')
    save_all_folds_to_csv(folds_data, folds_data_path)

    treeDic = loadTree(data_path, trainsetname)
    for fold_i in range(fold):
        test_var = 'fold' + str(fold_i) + '_test'
        train_var = 'fold' + str(fold_i) + '_train'
        train_losses, val_losses, train_accs, val_accs, accs, acc1, pre1, rec1, f1, acc2, pre2, rec2, f2 = train_GCN(
            treeDic,
            datasets[test_var], datasets[train_var], datasets['high_resource'],
            TDdroprate, BUdroprate, lr, weight_decay,
            patience, n_epochs, batchsize, trainsetname,
            iter, data_path, str(fold_i), alpha, beta, gamma,
            drop_edge_rate_1, drop_edge_rate_2,
            drop_feature_rate_1, drop_feature_rate_2, tau, alp)
        test_accs.append(accs)
        ACC1.append(acc1)
        ACC2.append(acc2)
        PRE1.append(pre1)
        PRE2.append(pre2)
        REC1.append(rec1)
        REC2.append(rec2)
        F1.append(f1)
        F2.append(f2)
print("Twitter:|Total_Test_ Accuracy: {:.4f}|acc1: {:.4f}|acc2: {:.4f}|pre1: {:.4f}|pre2: {:.4f}"
      "|rec1: {:.4f}|rec2: {:.4f}|F1: {:.4f}|F2: {:.4f}".format(st.mean(test_accs), st.mean(ACC1), st.mean(ACC2),
                                                                st.mean(PRE1),
                                                                st.mean(PRE2), st.mean(REC1), st.mean(REC2),
                                                                st.mean(F1), st.mean(F2)))
