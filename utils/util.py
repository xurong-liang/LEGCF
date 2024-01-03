import numpy as np
import os
import torch
from models.recsys import FM, BasicRecSys


def setup_seed(seed: int = 2020):
    """
    Set up random seed for this run
    """
    os.environ["PYTHONHASHSEED"] = str(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)
    np.random.seed(seed)
    torch.backends.cudnn.deterministic = True
    torch.backends.cudnn.enabled = True


def setup_recsys(*args, **kwargs) -> BasicRecSys:
    return FM(*args, **kwargs)


def print_opt(file_name: str, opt: dict):
    with open(file_name, "w") as fp:
        for key, val in opt.items():
            text = f"{key}: {val}"
            print(text)
            print(text, file=fp)
    print(f"param dict saved as {file_name}", flush=True)


def print_text(file_fp, text):
    print(text, file=file_fp, flush=True)
    print(text, flush=True)


def get_label(test_data, pred_data):
    """
    :param test_data: the collection of positively rated items in Test dataset for each user
    :param pred_data: the collection of items considered as positive by prediction
    :return test_data ranking & pred_data ranking
    """
    r = []
    for i in range(len(test_data)):
        groundTrue = test_data[i]
        predictTopK = pred_data[i]
        # == [_ in groundTrue for _ in predictTopK]
        pred = list(map(lambda x: x in groundTrue, predictTopK))
        pred = np.array(pred).astype("float")
        r.append(pred)
    return np.array(r).astype('float')


def ndcg_atk_r(test_data, r, k):
    """
    Normalized Discounted Cumulative Gain
    rel_i = 1 or 0, so 2^{rel_i} - 1 = 1 or 0
    """
    assert len(r) == len(test_data)
    pred_data = r[:, :k]

    test_matrix = np.zeros((len(pred_data), k))
    for i, items in enumerate(test_data):
        length = k if k <= len(items) else len(items)
        test_matrix[i, :length] = 1
    max_r = test_matrix
    idcg = np.sum(max_r * 1. / np.log2(np.arange(2, k + 2)), axis=1)
    dcg = pred_data * (1. / np.log2(np.arange(2, k + 2)))
    dcg = np.sum(dcg, axis=1)
    idcg[idcg == 0.] = 1.
    ndcg = dcg / idcg
    ndcg[np.isnan(ndcg)] = 0.
    return np.sum(ndcg)


def recall_atk(test_data, r, k):
    """
    test_data should be a list? cause users may have different amount of pos items. shape (test_batch, k)
    pred_data : shape (test_batch, k) NOTE: pred_data should be pre-sorted
    k : top-k

    """
    right_pred = r[:, :k].sum(1)
    recall_n = np.array([len(test_data[i]) for i in range(len(test_data))])
    correct_pred_count = right_pred / recall_n
    correct_pred_count[np.isnan(correct_pred_count)] = 0
    recall = np.sum(correct_pred_count)
    return recall


def save_model(recsys: FM, save_path: str):
    """
    Save model checkpoint.
    """
    # delete IO wrapper
    performance_fp, interact_mat = recsys.opt["performance_fp"], recsys.opt["interact_mat"]
    del recsys.opt["performance_fp"], recsys.opt["interact_mat"]

    # place model to cpu
    recsys.cpu()
    clustering = recsys.embedding.clustering
    clustering.centroid_assignment = clustering.centroid_assignment.cpu()
    clustering.expanded_norm_adj_graph = clustering.expanded_norm_adj_graph.cpu()

    if not save_path.endswith(".pt"):
        save_path += ".pt"
    torch.save(recsys, save_path)
    print(f"Model saved to {save_path}")

    # restore IO wrapper
    recsys.opt["performance_fp"], recsys.opt["interact_mat"] = performance_fp, interact_mat
    # place model back to original device
    recsys.to(recsys.opt["device_id"])
    clustering.centroid_assignment = clustering.centroid_assignment.to(recsys.opt["device_id"])
    clustering.expanded_norm_adj_graph = clustering.expanded_norm_adj_graph.to(recsys.opt["device_id"])


def load_model(opt: dict, load_path: str) -> FM:
    """
    Load model checkpoint
    """
    if not load_path.endswith(".pt"):
        load_path += ".pt"
    recsys: FM = torch.load(load_path, map_location="cpu")

    # replace the old opt with the current one
    recsys.opt = opt
    recsys.embedding.opt = opt
    recsys.embedding.clustering.opt = opt
    # load to correct device
    recsys = recsys.to(opt["device_id"])
    clustering = recsys.embedding.clustering
    clustering.centroid_assignment = clustering.centroid_assignment.coalesce().to(opt["device_id"])
    clustering.expanded_norm_adj_graph = clustering.expanded_norm_adj_graph.coalesce().to(opt["device_id"])

    print(f"Model loaded from {load_path}")
    return recsys
