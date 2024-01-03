"""
Script that implements all critical functions
"""
import torch
from models.recsys import BasicRecSys
from data_loading.data_loader import minibatch
import numpy as np
from utils.util import get_label, recall_atk, ndcg_atk_r
from data_loading.data_loader import DataLoader, generate_train_samples, shuffle
from parse import parse


def train_model(dataloader: DataLoader, recsys: BasicRecSys, optimizer: torch.optim.Optimizer, opt: dict):
    """
    We run a training epoch here.

    Train samples regenerated at each function call.

    :return avg_epoch_loss, avg_epoch_recsys_loss, avg_epoch_reg_loss
    """
    device_id = opt["device_id"]
    samples = generate_train_samples(dataloader=dataloader, neg_size=opt["neg_size"])
    users, pos_items, neg_items = torch.from_numpy(samples[:, 0]).long().to(device_id), \
        torch.from_numpy(samples[:, 1]).long().to(device_id), \
        torch.from_numpy(samples[:, 2]).long().to(device_id)
    users, pos_items, neg_items = shuffle(users, pos_items, neg_items)

    recsys.train()

    batch_count = len(users) // opt["bpr_batch"] + 1
    avg_epoch_loss, avg_epoch_recsys_loss, avg_epoch_reg_loss = 0., 0., 0.
    for idx, (batch_users, batch_pos, batch_neg) in enumerate(
            minibatch(users, pos_items, neg_items, batch_size=opt["bpr_batch"])):
        batch_users, batch_pos, batch_neg = batch_users.to(device_id), \
            batch_pos.to(device_id), batch_neg.to(device_id)
        opt["batch_idx"] = idx
        if opt["fixed_assignment_recsys_converged"] and idx == 0:
            # check if current epoch requires update
            opt["update_assignment"] = (
                    (opt["assignment_update_frequency"] == "every-epoch") or
                    (opt["assignment_update_frequency"] == "only-once" and opt.get(
                        "not_updated")))
            if not opt["update_assignment"]:
                # parse every-k-epochs
                frequency = int(parse("every-{}-epochs", opt["assignment_update_frequency"])[0])
                opt["update_assignment"] = opt["epoch_idx"] % frequency == 0
        else:
            opt["update_assignment"] = False
        batch_model_loss, batch_reg_loss = recsys.calculate_training_loss(batch_users, batch_pos, batch_neg)
        opt["update_assignment"] = False

        reg_decay = 1e-4
        batch_reg_loss = batch_reg_loss * reg_decay
        batch_loss = batch_model_loss + batch_reg_loss
        optimizer.zero_grad()
        batch_loss.backward()
        torch.nn.utils.clip_grad_value_(recsys.parameters(), opt["max_grad_value"])

        optimizer.step()

        avg_epoch_loss += batch_loss.item()
        avg_epoch_recsys_loss += batch_model_loss.item()

    avg_epoch_loss = avg_epoch_loss / batch_count
    avg_epoch_recsys_loss = avg_epoch_recsys_loss / batch_count
    avg_epoch_reg_loss = avg_epoch_reg_loss / batch_count
    return avg_epoch_loss, avg_epoch_recsys_loss, avg_epoch_reg_loss


def test_subprocess(X):
    """
    :param X: [(top k rating list, groundTrue_list)] for all users in current process
    """
    sorted_items = X[0].numpy()
    ground_true = X[1]
    # r: Test data ranking & top-k ranking
    r = get_label(ground_true, sorted_items)
    pre, recall, ndcg = [], [], []
    for k in ks:
        rec = recall_atk(ground_true, r, k)
        recall.append(rec)
        ndcg.append(ndcg_atk_r(ground_true, r, k))
    return {'recall': np.array(recall),
            'ndcg': np.array(ndcg)}


def pool_init_worker(opt: dict):
    global ks
    ks = opt["ks"]


def test_model(train_all_pos: dict, test_all_pos: dict, recsys: BasicRecSys, opt: dict):
    """
    Conduct testing on the model

    :param train_all_pos: key: uid, value: list of pos items used in train
    :param test_all_pos: key: uid, value: list of pos items in test dataset
    :return: {
        "ndcgs": [@5, @10, @20, @50 values],
        "recalls": [@5, @10, @20, @50 values]
    }
    """
    item_count = opt["field_dims"][1]
    test_batch_size = opt["test_batch"]
    recsys.eval()
    max_k = max(opt["ks"])
    # values @ k = 5, 10, 20, 50
    results = {
        "ndcgs": np.zeros(len(opt["ks"])),
        "recalls": np.zeros(len(opt["ks"]))
    }

    with torch.no_grad():
        users = list(range(opt['field_dims'][0]))
        try:
            assert test_batch_size <= len(users) / 10
        except AssertionError:
            print(f"test_u_batch_size is too big for this dataset, try a small one {len(users) // 10}")
        users_list = []
        rating_list = []
        ground_true_list = []

        batch_count = len(users) // test_batch_size + 1
        for idx, batch_users in enumerate(minibatch(users, batch_size=test_batch_size)):
            # the flag used to retrieve the full embedding for evaluating, will
            # be turned off so long as recsys() is called once
            opt["first_time_evaluating"] = idx == 0

            # all positive rated items by batch_users, list of lists of item IDs
            all_train_pos = [train_all_pos[_] for _ in batch_users]

            # ground true: all items positively rated in Test set
            ground_true = [test_all_pos.get(_, []) for _ in batch_users]
            batch_users = torch.tensor(batch_users).long().to(opt["device_id"])

            ratings = []
            for u in batch_users:
                u_repeated = torch.full((item_count,), u, device=opt["device_id"])
                u_rating = recsys(u_repeated, torch.arange(item_count, device=opt["device_id"]))
                ratings.append(u_rating)
            # rating dim: batch_user x all_items in the dataset
            ratings = torch.vstack(ratings)

            # exclude positively rated items in training step
            exclude_index = []
            exclude_items = []
            for _, items in enumerate(all_train_pos):
                exclude_index.extend([_] * len(items))
                exclude_items.extend(items)
            ratings[exclude_index, exclude_items] = -(1 << 10)

            # rating_k: max_k of items IDs that has the highest ranking
            _, rating_k = torch.topk(ratings, k=max_k)
            users_list.append(batch_users)
            rating_list.append(rating_k.cpu())
            ground_true_list.append(ground_true)
        assert batch_count == len(users_list) == len(rating_list) == len(ground_true_list)

        batch_process_input = zip(rating_list, ground_true_list)
        pool_init_worker(opt)
        computed_res = []
        for x in batch_process_input:
            computed_res.append(test_subprocess(x))

        for res in computed_res:
            results["recalls"] += res["recall"]
            results["ndcgs"] += res["ndcg"]
        results['recalls'] /= float(len(test_all_pos.keys()))
        results['ndcgs'] /= float(len(test_all_pos.keys()))
        results["recalls"], results["ndcgs"] = \
            results["recalls"].tolist(), results['ndcgs'].tolist()
    return results
