"""
The recsys used in final training.
"""

import torch
import torch.nn as nn
from torchfm.layer import FactorizationMachine, FeaturesLinear
from models.com_embedding import CompEmbedding, BasicEmbedding


def BPR_loss(pos_scores, neg_scores):
    dist = neg_scores - pos_scores
    return torch.mean(nn.functional.softplus(dist))


class BasicRecSys(nn.Module):
    """
    Skeleton of recommender system.
    """

    def __init__(self, opt, data_loader):
        super(BasicRecSys, self).__init__()
        self.data_loader = data_loader
        self.opt = opt
        self.latent_dim = opt['latent_dim']
        self.field_dims = opt['field_dims']
        self.feature_num = sum(self.field_dims)
        self.l2_penalty_factor = opt["l2_penalty_factor"]
        self.embedding = None

        # add this to item ids before embedding retrieval
        self.item_idx_offset = self.opt["field_dims"][0]

    def forward(self, uids, iids):
        """
        The forward function used for evaluation.
        """
        user_embs = self.embedding(uids)
        item_embs = self.embedding(iids + self.item_idx_offset)
        return user_embs, item_embs

    def calculate_training_loss(self, uids, pos_iids, neg_iids):
        """
        The loss calculation used for training.
        """
        user_embs = self.embedding(uids)
        pos_embs = self.embedding(pos_iids + self.item_idx_offset)
        neg_embs = self.embedding(neg_iids + self.item_idx_offset)
        return user_embs, pos_embs, neg_embs

    def reg_loss(self, user_embs, pos_embs, neg_embs):
        return self.l2_penalty_factor * (user_embs.norm(2).pow(2) +
                                         pos_embs.norm(2).pow(2) +
                                         neg_embs.norm(2).pow(2)) / len(user_embs)


class FM(BasicRecSys):
    """Factorization Machines Simple dot product. Use this for training and evaluation."""

    def __init__(self, opt, data_loader):
        super(FM, self).__init__(opt, data_loader)
        self.embedding: CompEmbedding = CompEmbedding(opt, data_loader)
        self.linear = FeaturesLinear(self.field_dims)  # linear part
        self.fm = FactorizationMachine(reduce_sum=True)

    def get_scores(self, uids, user_embs, iids, item_embs):
        fm_scores = self.fm(torch.concat((user_embs[:, None, :], item_embs[:, None, :]), dim=1)).squeeze()
        linear_scores = self.linear(torch.concat((uids[:, None], iids[:, None]), dim=1)).squeeze()
        return fm_scores, linear_scores

    def calculate_training_loss(self, uids, pos_iids, neg_iids):
        """
        :return batch bpr loss, reg loss
        """
        ego_embs, gcn_embs = self.embedding.update()
        idxes = torch.cat((uids, pos_iids + self.item_idx_offset, neg_iids + self.item_idx_offset))
        batch_gcn_embs, batch_ego_embs = gcn_embs[idxes], ego_embs[idxes]
        user_embs = batch_gcn_embs[:len(uids)]
        user_ego_embs = batch_ego_embs[:len(uids)]
        pos_embs = batch_gcn_embs[len(uids): len(uids) + len(pos_iids)]
        pos_ego_embs = batch_ego_embs[len(uids): len(uids) + len(pos_iids)]
        neg_embs = batch_gcn_embs[len(uids) + len(pos_iids):]
        neg_ego_embs = batch_ego_embs[len(uids) + len(pos_iids):]

        pos_fm_scores, pos_linear_scores = self.get_scores(uids, user_embs, pos_iids, pos_embs)
        pos_scores = pos_fm_scores + pos_linear_scores
        assert not torch.isnan(pos_fm_scores).any()
        assert not torch.isnan(pos_linear_scores).any()
        neg_fm_scores, neg_linear_scores = self.get_scores(uids, user_embs, neg_iids, neg_embs)
        assert not torch.isnan(neg_fm_scores).any()
        assert not torch.isnan(neg_linear_scores).any()
        neg_scores = neg_fm_scores + neg_linear_scores
        bpr_loss = BPR_loss(pos_scores, neg_scores)
        assert not torch.isnan(bpr_loss).any() and not torch.isinf(bpr_loss).any()
        reg_loss = self.reg_loss(user_embs=user_ego_embs, pos_embs=pos_ego_embs, neg_embs=neg_ego_embs)
        return bpr_loss, reg_loss

    def forward(self, uids, iids):
        idxes = torch.cat((uids, iids + self.item_idx_offset))
        gcn_embs = self.embedding(idxes)
        user_embs = gcn_embs[:len(uids)]
        item_embs = gcn_embs[len(uids):]
        fm_scores, linear_scores = self.get_scores(uids, user_embs, iids, item_embs)
        return fm_scores + linear_scores
