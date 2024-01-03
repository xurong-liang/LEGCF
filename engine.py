import os
import torch.optim
from data_loading.data_loader import DataLoader
from utils.util import setup_seed, setup_recsys, print_opt, print_text, save_model, load_model
from set_parse import config_parser
from tensorboardX import SummaryWriter
from timeit import default_timer as timer
from datetime import timedelta
from funcs import train_model, test_model
import numpy as np
import scipy.sparse as sp
from parse import parse

torch.autograd.set_detect_anomaly(True)
np.seterr(divide="ignore")


def run():
    global recsys
    early_stop_count = None if opt["early_stop_fixed_assignment_stage"] == -1 else 0
    pre_best_ndcg_5 = pre_best_ndcg_epoch = post_best_ndcg_5 = post_best_ndcg_epoch = None

    for epoch_idx in range(opt["epochs"]):
        if opt["early_stop_fixed_assignment_stage"] and opt["early_stop_assignment_update_stage"]:
            early_stop_target = opt["early_stop_fixed_assignment_stage"] \
                if not opt["fixed_assignment_recsys_converged"] else opt["early_stop_assignment_update_stage"]
        else:
            early_stop_target = None

        # evaluate here
        if epoch_idx % 1 == 0:
            # evaluate every 10 epochs
            print("[Validation]", flush=True)
            evaluated_start = timer()
            # here test all pos is validation set
            result_dict = test_model(train_all_pos=data_loader.train_all_pos, test_all_pos=data_loader.valid_all_pos,
                                     recsys=recsys, opt=opt)
            evaluated_end = timer()
            out_text = f"Epoch {epoch_idx:d} | ndcgs: {result_dict['ndcgs']} | recalls: {result_dict['recalls']}"
            print_text(file_fp=performance_fp, text=out_text)
            print(f"Time used  = {timedelta(seconds=evaluated_end - evaluated_start)}.", flush=True)

            for _idx in range(len(opt["ks"])):
                k_val = opt["ks"][_idx]
                writer.add_scalar(f"test/epoch_wise/ndcg_{k_val:d}", result_dict["ndcgs"][_idx], epoch_idx)
                writer.add_scalar(f"test/epoch_wise/recall_{k_val:d}", result_dict["recalls"][_idx], epoch_idx)

            current_ndcg_5 = result_dict["ndcgs"][0]

            if not opt["fixed_assignment_recsys_converged"]:
                best_ndcg_5 = pre_best_ndcg_5
                checkpoint_save_path = os.path.join(opt["res_prepath"], "best_checkpoint_before_assignment_update")
            else:
                best_ndcg_5 = post_best_ndcg_5
                checkpoint_save_path = os.path.join(opt["res_prepath"], "best_checkpoint_with_assignment_update")

            if best_ndcg_5 is None or current_ndcg_5 > best_ndcg_5:
                # reset early stop
                if early_stop_count is not None:
                    early_stop_count = 0
                    # save model checkpoint
                    save_model(recsys, checkpoint_save_path)
                    print(f"Epoch {epoch_idx}: best ndcg@5 change from {best_ndcg_5} "
                          f"to {current_ndcg_5}. Model saved.", flush=True)
                    if not opt["fixed_assignment_recsys_converged"]:
                        pre_best_ndcg_5, pre_best_ndcg_epoch = current_ndcg_5, epoch_idx
                    else:
                        post_best_ndcg_5, post_best_ndcg_epoch = current_ndcg_5, epoch_idx
            elif early_stop_count is not None:
                # increment early stop
                early_stop_count += 1
            print(f"early stop flag: {early_stop_count}/{early_stop_target}, fixed assignment training converged: "
                  f"{opt['fixed_assignment_recsys_converged']}.", flush=True)

        if early_stop_count is not None and early_stop_count == early_stop_target:
            if opt['fixed_assignment_recsys_converged']:
                text = f"early stop threshold {early_stop_count} reached. Now exit."
                print_text(file_fp=opt["performance_fp"], text=text)
                break
            else:
                text = "fixed assignment training converged. Now early stop flag reset."
                print_text(file_fp=opt["performance_fp"], text=text)
                opt['fixed_assignment_recsys_converged'] = True
                early_stop_count = 0

        # train here
        train_start = timer()
        opt["epoch_idx"] = epoch_idx
        all_loss = train_model(dataloader=data_loader, recsys=recsys, optimizer=optimizer, opt=opt)
        avg_epoch_loss, avg_epoch_recsys_loss, avg_epoch_reg_loss = all_loss
        train_end = timer()

        writer.add_scalar("train/epoch_wise/avg_epoch_loss", avg_epoch_loss, epoch_idx)
        writer.add_scalar("train/epoch_wise/avg_epoch_recsys_loss", avg_epoch_recsys_loss, epoch_idx)
        writer.add_scalar("train/epoch_wise/avg_epoch_reg_loss", avg_epoch_reg_loss, epoch_idx)
        text = f"[Epoch {epoch_idx}/{opt['epochs'] - 1}] avg epoch loss = {avg_epoch_loss:.4f} |" \
               f" avg recsys loss = {avg_epoch_recsys_loss:.4f} | avg reg loss = {avg_epoch_reg_loss:.8f} " \
               f"\ntime elapsed = {timedelta(seconds=train_end - train_start)}."
        print(text, flush=True)
    out_text = f"\nBest pre-ndcg@5 at Epoch {pre_best_ndcg_epoch}, Best post-ndcg@5 at Epoch {post_best_ndcg_epoch}.\n\n"
    print_text(file_fp=performance_fp, text=out_text)

    # reach here we evaluate the best model
    print_text(file_fp=performance_fp, text="[Evaluation]")
    for model_name in ["best_checkpoint_before_assignment_update", "best_checkpoint_with_assignment_update"]:
        print_text(file_fp=performance_fp, text=f"Evaluating {model_name}...")
        checkpoint_save_path = os.path.join(opt["res_prepath"], f"{model_name}.pt")
        recsys = load_model(opt, checkpoint_save_path)
        # save best centroid embs
        centroid_file_name = os.path.join(recsys.embedding.clustering.centroid_save_path,
                                          f"{model_name}.npz")
        sp.save_npz(centroid_file_name,
                    sp.csr_matrix(recsys.embedding.clustering.centroid_embs.data.detach().cpu().numpy()))
        evaluated_start = timer()
        result_dict = test_model(train_all_pos=data_loader.train_all_pos, test_all_pos=data_loader.test_all_pos,
                                 recsys=recsys, opt=opt)
        evaluated_end = timer()
        out_text = f"ndcgs: {result_dict['ndcgs']} | recalls: {result_dict['recalls']}\n\n"
        print_text(file_fp=performance_fp, text=out_text)
        print(f"Time used  = {timedelta(seconds=evaluated_end - evaluated_start)}.\n\n", flush=True)


if __name__ == "__main__":
    parser = config_parser()
    opt = vars(parser.parse_args())
    setup_seed(opt["seed"])
    if torch.cuda.is_available():
        if not isinstance(opt['device_id'], int) and opt["device_id"].isnumeric():
            opt["device_id"] = f"cuda:{opt['device_id']}"
    else:
        opt["device_id"] = "cpu"

    opt["data_path"] = f"./data/{opt['dataset_name']}"

    opt["alias"] = f"{opt['dataset_name']}_latent_dim_{opt['latent_dim']:d}_" \
                   f"num_cluster_{opt['num_clusters']}_seed_{opt['seed']:d}_num_composition_embs_{opt['num_composition_centroid']}" \
                   f"_lr_{opt['lr']:.0e}_optimizer_weight_decay_{opt['optimizer_weight_decay']}_" \
                   f"l2_penalty_{opt['l2_penalty_factor']}"
    if opt["additional_alias"]:
        opt["alias"] += "_" + opt["additional_alias"]

    if opt["assignment_update_frequency"] not in ["every-epoch", "only-once"]:
        # check if it is in form "every-{}-epochs"
        frequency = parse("every-{}-epochs", opt["assignment_update_frequency"])
        if not frequency:
            print("update frequency should be the form: only-once, every-epoch, every-k-epochs")
            exit(2)
        elif int(frequency[0]) < 1:
            print("invalid update frequency")
            exit(2)

    opt["res_prepath"] = os.path.join(opt["res_prepath"], opt["alias"])

    if not opt.get("num_clusters"):
        print("Please specify the number of clusters to be created.")
    os.makedirs(opt["res_prepath"], exist_ok=True)

    # get dataloader
    data_loader = DataLoader(opt=opt)
    interact_mat = data_loader.interact_mat
    opt["field_dims"] = list(interact_mat.shape)

    opt_out_dir = os.path.join(opt["res_prepath"], "params.txt")
    print_opt(file_name=opt_out_dir, opt=opt)

    opt["interact_mat"] = interact_mat

    # performance output
    performance_fp_path = os.path.join(opt["res_prepath"], "res.txt")

    performance_fp = open(performance_fp_path, "w")
    opt["performance_fp"] = performance_fp

    # recsys, optimizer, train datasets
    recsys = setup_recsys(opt, data_loader).to(opt["device_id"])
    opt["fixed_assignment_recsys_converged"] = False

    optimizer = torch.optim.Adam(recsys.parameters(), lr=opt["lr"], weight_decay=opt["optimizer_weight_decay"])

    # tensorboard
    os.makedirs(os.path.join(opt["res_prepath"], "runs"), exist_ok=True)
    tensor_board_path = os.path.join(opt["res_prepath"], "runs", opt["alias"])
    writer = SummaryWriter(tensor_board_path)
    writer.add_text("option", str(opt), 0)

    if torch.cuda.is_available():
        torch.cuda.empty_cache()

    time_start = timer()
    # save init model
    init_save_path = os.path.join(opt["res_prepath"], "init_checkpoint")
    # save model checkpoint
    save_model(recsys, init_save_path)
    try:
        run()
    except Exception as e:
        raise e
    finally:
        writer.close()
        performance_fp.close()
    time_end = timer()
    print(f"Program ends. Time elapsed = {timedelta(seconds=time_end - time_start)}.", flush=True)
