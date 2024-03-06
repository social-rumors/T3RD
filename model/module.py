import torch as th
import copy
import random
import torch.nn.functional as F
import torch_geometric
from torch_scatter import scatter_mean
from torch_geometric.nn import GCNConv
from Process.distribution_Alignment import covariance, coral, linear_mmd
from Process.contrastive import GCN, AvgReadout, Discriminator, Encoder
from Process.process import sparse_mx_to_torch_sparse_tensor, get_activation, get_base_model
from typing import Optional

device = th.device('cuda:0' if th.cuda.is_available() else 'cpu')


class TDrumorGCN(th.nn.Module):
    def __init__(self, in_feats, hid_feats, out_feats):
        super(TDrumorGCN, self).__init__()
        self.conv1 = GCNConv(in_feats, hid_feats)
        self.conv2 = GCNConv(hid_feats + in_feats, out_feats)

    def forward(self, x, edge_index, data):
        x1 = copy.copy(x.float())
        x = self.conv1(x, edge_index)
        x2 = copy.copy(x)
        rootindex = data.rootindex
        root_extend = th.zeros(len(data.batch), x1.size(1)).to(device)
        batch_size = max(data.batch) + 1
        for num_batch in range(batch_size):
            index = (th.eq(data.batch, num_batch))
            root_extend[index] = x1[rootindex[num_batch]]
        x = th.cat((x, root_extend), 1)

        x = F.relu(x)
        x = F.dropout(x, training=self.training)
        x = self.conv2(x, edge_index)
        x = F.relu(x)
        root_extend = th.zeros(len(data.batch), x2.size(1)).to(device)
        for num_batch in range(batch_size):
            index = (th.eq(data.batch, num_batch))
            root_extend[index] = x2[rootindex[num_batch]]
        x = th.cat((x, root_extend), 1)

        return x


class BUrumorGCN(th.nn.Module):
    def __init__(self, in_feats, hid_feats, out_feats):
        super(BUrumorGCN, self).__init__()
        self.conv1 = GCNConv(in_feats, hid_feats)
        self.conv2 = GCNConv(hid_feats + in_feats, out_feats)

    def forward(self, x, edge_index, data):
        x1 = copy.copy(x.float())
        x = self.conv1(x, edge_index)
        x2 = copy.copy(x)

        rootindex = data.rootindex
        root_extend = th.zeros(len(data.batch), x1.size(1)).to(device)
        batch_size = max(data.batch) + 1
        for num_batch in range(batch_size):
            index = (th.eq(data.batch, num_batch))
            root_extend[index] = x1[rootindex[num_batch]]
        x = th.cat((x, root_extend), 1)

        x = F.relu(x)
        x = F.dropout(x, training=self.training)
        x = self.conv2(x, edge_index)
        x = F.relu(x)
        root_extend = th.zeros(len(data.batch), x2.size(1)).to(device)
        for num_batch in range(batch_size):
            index = (th.eq(data.batch, num_batch))
            root_extend[index] = x2[rootindex[num_batch]]
        x = th.cat((x, root_extend), 1)

        return x


class SCL(th.nn.Module):
    def __init__(self, temperature=0.1):
        super(SCL, self).__init__()
        self.temperature = temperature

    def forward(self, inrep_1, inrep_2, label_1, label_2=None):
        inrep_1.to(device)
        inrep_2.to(device)
        bs_1 = int(inrep_1.shape[0])
        bs_2 = int(inrep_2.shape[0])

        if label_2 == None:
            normalize_inrep_1 = F.normalize(inrep_1, p=2, dim=1)
            normalize_inrep_2 = F.normalize(inrep_2, p=2, dim=1)
            cosine_similarity = th.matmul(normalize_inrep_1, normalize_inrep_2.t())  # bs_1, bs_2

            diag = th.diag(cosine_similarity)
            cos_diag = th.diag_embed(diag)  # bs,bs

            label = th.unsqueeze(label_1, -1)
            if label.shape[0] == 1:
                cos_loss = th.zeros(1)
            else:
                for i in range(label.shape[0] - 1):
                    if i == 0:
                        label_mat = th.cat((label, label), -1)
                    else:
                        label_mat = th.cat((label_mat, label), -1)  # bs, bs

                mid_mat_ = (label_mat.eq(label_mat.t()))
                mid_mat = mid_mat_.float()

                cosine_similarity = (cosine_similarity - cos_diag) / self.temperature  # the diag is 0
                mid_diag = th.diag_embed(th.diag(mid_mat))
                mid_mat = mid_mat - mid_diag

                cosine_similarity = cosine_similarity.masked_fill_(mid_diag.byte(), -float('inf'))  # mask the diag

                cos_loss = th.log(
                    th.clamp(F.softmax(cosine_similarity, dim=1) + mid_diag, 1e-10, 1e10))  # the sum of each row is 1

                cos_loss = cos_loss * mid_mat

                cos_loss = th.sum(cos_loss, dim=1) / (th.sum(mid_mat, dim=1) + 1e-10)  # bs

        else:
            if bs_1 != bs_2:
                while bs_1 < bs_2:
                    inrep_2 = inrep_2[:bs_1]
                    label_2 = label_2[:bs_1]
                    break
                while bs_2 < bs_1:
                    inrep_2_ = inrep_2
                    ra = random.randint(0, int(inrep_2_.shape[0]) - 1)
                    pad = inrep_2_[ra].unsqueeze(0)
                    lbl_pad = label_2[ra].unsqueeze(0)
                    inrep_2 = th.cat((inrep_2, pad), 0)
                    label_2 = th.cat((label_2, lbl_pad), 0)
                    bs_2 = int(inrep_2.shape[0])

            normalize_inrep_1 = F.normalize(inrep_1, p=2, dim=1)
            normalize_inrep_2 = F.normalize(inrep_2, p=2, dim=1)
            cosine_similarity = th.matmul(normalize_inrep_1, normalize_inrep_2.t())  # bs_1, bs_2

            label_1 = th.unsqueeze(label_1, -1)
            label_1_mat = th.cat((label_1, label_1), -1)
            for i in range(label_1.shape[0] - 1):
                if i == 0:
                    label_1_mat = label_1_mat
                else:
                    label_1_mat = th.cat((label_1_mat, label_1), -1)  # bs, bs

            label_2 = th.unsqueeze(label_2, -1)
            label_2_mat = th.cat((label_2, label_2), -1)
            for i in range(label_2.shape[0] - 1):
                if i == 0:
                    label_2_mat = label_2_mat
                else:
                    label_2_mat = th.cat((label_2_mat, label_2), -1)  # bs, bs

            mid_mat_ = (label_1_mat.t().eq(label_2_mat))
            mid_mat = mid_mat_.float()

            cosine_similarity = cosine_similarity / self.temperature
            cos_loss = th.log(th.clamp(F.softmax(cosine_similarity, dim=1), 1e-10, 1e10))
            cos_loss = cos_loss * mid_mat  # find the sample with the same label
            cos_loss = th.sum(cos_loss, dim=1) / th.sum(mid_mat + 1e-10, dim=1)  # bs

        cos_loss = -th.mean(cos_loss, dim=0)
        return cos_loss


class GlobalContrastiveLearning(th.nn.Module):
    def __init__(self, n_in, n_h, n_out, activation):
        super(GlobalContrastiveLearning, self).__init__()
        self.gcn = GCN(n_h + n_out, n_h, activation)
        self.read = AvgReadout()
        self.sigm = th.nn.Sigmoid()
        self.disc = Discriminator(n_h)

    def forward(self, edge_index, raw_view, aug1_view):
        raw_view.to(device)
        aug1_view.to(device)
        edge_index.to(device)
        adj = torch_geometric.utils.to_scipy_sparse_matrix(edge_index)
        nb_nodes = raw_view.shape[0]
        msk, samp_bias1, samp_bias2 = None, None, None
        sp_adj = sparse_mx_to_torch_sparse_tensor(adj)
        lbl_1 = th.ones(1, nb_nodes)
        lbl_2 = th.zeros(1, nb_nodes)
        lbl = th.cat((lbl_1, lbl_2), 1)
        if th.cuda.is_available():
            raw_view = raw_view.to(device)
            sp_adj = sp_adj.to(device)
            lbl = lbl.to(device)
        h_1 = self.gcn(raw_view, sp_adj, True)
        c = self.read(h_1, msk)
        c = self.sigm(c)
        h_2 = self.gcn(aug1_view, sp_adj, True)
        ret = self.disc(c, h_1, h_2, samp_bias1, samp_bias2)
        b_xent = th.nn.BCEWithLogitsLoss()
        loss = b_xent(ret, lbl)
        return loss


class LocalContrastiveLearning(th.nn.Module):
    def __init__(self, encoder: Encoder, num_hidden: int, num_proj_hidden: int, tau: float):
        super(LocalContrastiveLearning, self).__init__()
        self.encoder: Encoder = encoder
        self.tau: float = tau

        self.fc1 = th.nn.Linear(num_hidden, num_proj_hidden)
        self.fc2 = th.nn.Linear(num_proj_hidden, num_hidden)

        self.num_hidden = num_hidden

    def forward(self, x: th.Tensor, edge_index: th.Tensor) -> th.Tensor:
        return self.encoder(x, edge_index)

    def projection(self, z: th.Tensor) -> th.Tensor:
        z = F.elu(self.fc1(z))
        return self.fc2(z)

    def get_Id(self, z: th.Tensor):
        I = th.eye(z.shape[0]).to(z.device)
        Id = (th.norm((z @ z.T) - I, p='fro').pow(2))
        return Id

    def sim(self, z1: th.Tensor, z2: th.Tensor):
        z1 = F.normalize(z1)
        z2 = F.normalize(z2)
        return th.mm(z1, z2.t())

    def semi_loss(self, z1: th.Tensor, z2: th.Tensor):
        f = lambda x: th.exp(x / self.tau)
        refl_sim = f(self.sim(z1, z1))
        between_sim = f(self.sim(z1, z2))

        return -th.log(between_sim.diag() / (refl_sim.sum(1) + between_sim.sum(1) - refl_sim.diag() + (1e-10)))

    def batched_semi_loss(self, z1: th.Tensor, z2: th.Tensor, batch_size: int):
        # Space complexity: O(BN) (semi_loss: O(N^2))
        device = z1.device
        num_nodes = z1.size(0)
        num_batches = (num_nodes - 1) // batch_size + 1
        f = lambda x: th.exp(x / self.tau)
        indices = th.arange(0, num_nodes).to(device)
        losses = []

        for i in range(num_batches):
            mask = indices[i * batch_size:(i + 1) * batch_size]
            refl_sim = f(self.sim(z1[mask], z1))  # [B, N]
            between_sim = f(self.sim(z1[mask], z2))  # [B, N]

            losses.append(-th.log(between_sim[:, i * batch_size:(i + 1) * batch_size].diag()
                                  / (refl_sim.sum(1) + between_sim.sum(1)
                                     - refl_sim[:, i * batch_size:(i + 1) * batch_size].diag() + (1e-10))))

        return th.cat(losses)

    def loss(self, z1: th.Tensor, z2: th.Tensor, mean: bool = True, batch_size: Optional[int] = None):
        h1 = self.projection(z1)
        h2 = self.projection(z2)

        if batch_size is None:
            l1 = self.semi_loss(h1, h2)
            l2 = self.semi_loss(h2, h1)
        else:
            l1 = self.batched_semi_loss(h1, h2, batch_size)
            l2 = self.batched_semi_loss(h2, h1, batch_size)

        ret = (l1 + l2) * 0.5
        ret = ret.mean() if mean else ret.sum()

        return ret


class Net(th.nn.Module):
    def __init__(self, in_feats, hid_feats, out_feats, temperature, activation):
        super(Net, self).__init__()
        self.TDrumorGCN = TDrumorGCN(in_feats, hid_feats, out_feats)
        self.BUrumorGCN = BUrumorGCN(in_feats, hid_feats, out_feats)
        self.fc = th.nn.Linear((out_feats + hid_feats) * 2, 2)
        self.scl = SCL(temperature)
        self.gcl = GlobalContrastiveLearning(in_feats, hid_feats, out_feats, activation)
        self.lcl = LocalContrastiveLearning(Encoder(in_feats, hid_feats, get_activation(activation),
                                                    base_model=get_base_model('GCNConv'), k=2), hid_feats, out_feats,
                                            temperature)

    def forward(self, flag, alpha, beta, gamma, data, twitter_data=None, feat_cov_TD=None, feat_mean_TD=None, alp=0.1):
        if twitter_data is None:
            if flag == 'test-time':
                # self-supervised auxiliary task - Global contrast_TD
                SSL_TD_x = data.x
                SSL_TD_aug1_x = data.aug1_x
                SSL_TD_edge_index = data.ori_edge_index
                SSL_TD_raw_y = self.TDrumorGCN(SSL_TD_x, SSL_TD_edge_index, data)
                SSL_TD_aug1_y = self.TDrumorGCN(SSL_TD_aug1_x, SSL_TD_edge_index, data)
                gcl_TD_loss = self.gcl(SSL_TD_edge_index, SSL_TD_raw_y, SSL_TD_aug1_y)

                # self-supervised auxiliary task - Local contrast_TD
                SSL_TD_aug2_x = data.aug2_x
                SSL_TD_aug3_x = data.aug3_x
                SSL_TD_aug2_edge_index = data.aug2_edge_index
                SSL_TD_aug3_edge_index = data.aug3_edge_index
                SSL_TD_aug2_y = self.TDrumorGCN(SSL_TD_aug2_x, SSL_TD_aug2_edge_index, data)
                SSL_TD_aug3_y = self.TDrumorGCN(SSL_TD_aug3_x, SSL_TD_aug3_edge_index, data)
                SSL_TD_aug2_z = self.lcl(SSL_TD_aug2_y, SSL_TD_aug2_edge_index)
                SSL_TD_aug3_z = self.lcl(SSL_TD_aug3_y, SSL_TD_aug3_edge_index)
                lcl_TD_loss = self.lcl.loss(SSL_TD_aug2_z, SSL_TD_aug3_z, batch_size=1024)

                # feature alignment
                test_feat_stack_TD = [SSL_TD_raw_y, SSL_TD_aug1_y, SSL_TD_aug2_y, SSL_TD_aug3_y]
                test_feat_all_TD = th.cat(test_feat_stack_TD)
                test_feat_cov_TD = covariance(test_feat_all_TD)
                test_feat_mean_TD = test_feat_all_TD.mean(dim=0)
                loss_cov_TD = coral(feat_cov_TD, test_feat_cov_TD)
                loss_mm_TD = linear_mmd(feat_mean_TD, test_feat_mean_TD)
                ada_loss_TD = loss_cov_TD + loss_mm_TD

                total_loss = (gcl_TD_loss + beta * lcl_TD_loss) + gamma * ada_loss_TD
                return total_loss

            elif flag == 'test':
                x = data.x
                seq_len = data.seqlen
                edge_index = data.edge_index
                BU_edge_index = data.BU_edge_index

                TD_x = self.TDrumorGCN(x, edge_index, data)
                TD_x = scatter_mean(TD_x, data.batch, dim=0)
                BU_x = self.BUrumorGCN(x, BU_edge_index, data)
                BU_x = scatter_mean(BU_x, data.batch, dim=0)

                x = th.cat((BU_x, TD_x), 1)
                x = self.fc(x)
                x = F.log_softmax(x, dim=1)
                return x
        else:
            alp = alp
            t = twitter_data.x
            TD_t = self.TDrumorGCN(t, twitter_data.edge_index, twitter_data)
            TD_t = scatter_mean(TD_t, twitter_data.batch, dim=0)
            BU_t = self.BUrumorGCN(t, twitter_data.BU_edge_index, twitter_data)
            BU_t = scatter_mean(BU_t, twitter_data.batch, dim=0)
            t_ = th.cat((BU_t, TD_t), 1)
            twitter_scloss = self.scl(t_, t_, twitter_data.y)
            t = self.fc(t_)
            t = F.log_softmax(t, dim=1)
            twitter_CEloss = F.nll_loss(t, twitter_data.y)

            x = data.x
            TD_x = self.TDrumorGCN(x, data.edge_index, data)
            TD_x = scatter_mean(TD_x, data.batch, dim=0)
            BU_x = self.BUrumorGCN(x, data.BU_edge_index, data)
            BU_x = scatter_mean(BU_x, data.batch, dim=0)
            x_ = th.cat((BU_x, TD_x), 1)
            weibocovid19_scloss = self.scl(x_, t_, data.y, twitter_data.y)
            x = self.fc(x_)
            x = F.log_softmax(x, dim=1)
            weibocovid19_CEloss = F.nll_loss(x, data.y)

            main_loss = (((1 - alp) * twitter_CEloss + alp * twitter_scloss) +
                         ((1 - alp) * weibocovid19_CEloss + alp * weibocovid19_scloss))

            # high-resource auxiliary task
            # self-supervised auxiliary task - Global contrast_TD
            high_SSL_TD_x = twitter_data.x
            high_SSL_TD_aug1_x = twitter_data.aug1_x
            high_SSL_TD_edge_index = twitter_data.ori_edge_index
            high_SSL_TD_raw_y = self.TDrumorGCN(high_SSL_TD_x, high_SSL_TD_edge_index, twitter_data)
            high_SSL_TD_aug1_y = self.TDrumorGCN(high_SSL_TD_aug1_x, high_SSL_TD_edge_index, twitter_data)
            high_gcl_TD_loss = self.gcl(high_SSL_TD_edge_index, high_SSL_TD_raw_y, high_SSL_TD_aug1_y)

            # self-supervised auxiliary task - Local contrast_TD
            high_SSL_TD_aug2_x = twitter_data.aug2_x
            high_SSL_TD_aug3_x = twitter_data.aug3_x
            high_SSL_TD_aug2_edge_index = twitter_data.aug2_edge_index
            high_SSL_TD_aug3_edge_index = twitter_data.aug3_edge_index
            high_SSL_TD_aug2_y = self.TDrumorGCN(high_SSL_TD_aug2_x, high_SSL_TD_aug2_edge_index, twitter_data)
            high_SSL_TD_aug3_y = self.TDrumorGCN(high_SSL_TD_aug3_x, high_SSL_TD_aug3_edge_index, twitter_data)
            high_SSL_TD_aug2_z = self.lcl(high_SSL_TD_aug2_y, high_SSL_TD_aug2_edge_index)
            high_SSL_TD_aug3_z = self.lcl(high_SSL_TD_aug3_y, high_SSL_TD_aug3_edge_index)
            high_lcl_TD_loss = self.lcl.loss(high_SSL_TD_aug2_z, high_SSL_TD_aug3_z, batch_size=1024)

            total_loss = main_loss + alpha * (high_gcl_TD_loss + beta * high_lcl_TD_loss)
            return total_loss, x
