from data_loading.data_loader import DataLoader
import torch
import torch.nn as nn
from models.mat_approx import MatApprox


class BasicEmbedding(nn.Module):
    """
    Full embedding format for pretraining.
    """

    def __init__(self, opt, data_loader: DataLoader):
        super(BasicEmbedding, self).__init__()
        self.opt = opt
        self.latent_dim = self.opt["latent_dim"]

        self.embedding_module = nn.Parameter(torch.rand(sum(self.opt['field_dims']), self.latent_dim))
        nn.init.xavier_uniform_(self.embedding_module)

    def forward(self, offset_ids):
        """
        :param offset_ids: ids with index offset added (if applicable)
        :return requested entity embeddings
        """
        return self.embedding_module[offset_ids, :]

    def save_pretrain_embedding(self, file_name: str):
        torch.save(self.embedding_module.data.cpu(), file_name)
        print(f"Pretrain embeddings saved to {file_name}")


class CompEmbedding(BasicEmbedding):
    """
    Compositional embedding.
    """

    def __init__(self, opt, data_loader: DataLoader):
        super(CompEmbedding, self).__init__(opt, data_loader)

        # embedding table is a SpectralClustering instance
        self.embedding_module = None
        self.clustering = MatApprox(opt=opt, data_loader=data_loader)

        # full LightGCN embedding used for evaluation
        self.h_full_embs_for_evaluation = None

    def update(self):
        """
        Update cluster assignment, centroid embeddings
        """
        # centroid aggregation
        ego_embs, gcn_embs = self.clustering.forward(mode="train")
        return ego_embs, gcn_embs

    def forward(self, offset_ids):
        """
        Do not update centroid embedding here. For evaluation use only.

        :param offset_ids: ids with index offset added (if applicable)
        :return requested GCN embeddings
        """
        if self.opt["first_time_evaluating"]:
            # retrieve full embedding
            self.h_full_embs_for_evaluation = self.clustering.forward("test")
            # turn off flag for obtain full embs immediately
            self.opt["first_time_evaluating"] = False
        gcn_embs = self.h_full_embs_for_evaluation
        return gcn_embs[offset_ids]

