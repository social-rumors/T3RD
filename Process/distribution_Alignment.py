import torch
import statistics


def covariance(features):
    # assert len(features.size()) == 2, "TODO: multi-dimensional feature map covariance"
    n = features.shape[0]
    tmp = torch.ones((1, n), device=features.device) @ features
    cov = (features.t() @ features - (tmp.t() @ tmp) / n) / (n - 1)
    return cov


def coral(cs, ct):
    d = cs.shape[0]
    loss = (cs - ct).pow(2).sum() / (4. * d ** 2)
    return loss


def linear_mmd(ms, mt):
    loss = (ms - mt).pow(2).mean()
    return loss


def offline(train_loader, model_shared, device):
    mu_src = None
    cov_src = None
    coral_stack = []
    mmd_stack = []
    feat_stack = []
    with torch.no_grad():
        for train_data in train_loader:
            train_data.to(device)
            feat = model_shared(train_data.x, train_data.edge_index, train_data)
            cov = covariance(feat)
            mu = feat.mean(dim=0)
            if cov_src is None:
                cov_src = cov
                mu_src = mu
            else:
                loss_coral = coral(cov_src, cov)
                loss_mmd = linear_mmd(mu_src, mu)
                coral_stack.append(loss_coral.item())
                mmd_stack.append(loss_mmd.item())
                feat_stack.append(feat)
    feat_all = torch.cat(feat_stack)
    feat_cov = covariance(feat_all)
    feat_mean = feat_all.mean(dim=0)
    statistics_coral_mean = statistics.mean(coral_stack)
    statistics_mmd_mean = statistics.mean(mmd_stack)
    return feat_cov, statistics_coral_mean, feat_mean, statistics_mmd_mean
