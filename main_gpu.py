
import time
import torch.nn as nn
import numpy as np
import torch
from data.dataset_gpu import DyDataset
from models.progressive_sage_gpu import GraphSage
from utils.get_params import get_args
from utils.device import get_device
from utils.set_seed import set_random_seed
from utils.causal_machanism_gpu import generate_causal_labels
import warnings
import os
import copy
import hashlib
warnings.filterwarnings('ignore')
from models.meta_loss import BackwardTransferDynamicGNN
from sklearn.cluster import KMeans
from sklearn.metrics.cluster import normalized_mutual_info_score
from sklearn.metrics.cluster import adjusted_rand_score
from models.non_overlapping_community_model import non_overlapping_main
from selective_modeling.modules.graph_selective_modeling import Mamba
import torch.nn.functional as F



class SparseEnvAnchorMemory(nn.Module):
    def __init__(self, hidden_dim):
        super().__init__()
        self.hidden_dim = hidden_dim
        self.query_proj = nn.Linear(hidden_dim, hidden_dim, bias=False)
        self.key_proj = nn.Linear(hidden_dim, hidden_dim, bias=False)
        self.gate_s = nn.Linear(hidden_dim, hidden_dim, bias=False)
        self.gate_z = nn.Linear(hidden_dim, hidden_dim, bias=False)
        self.gate_b = nn.Parameter(torch.zeros(hidden_dim))
        self.out_proj = nn.Linear(hidden_dim * 2, hidden_dim, bias=False)
        self._reset_parameters()

    def _reset_parameters(self):
        with torch.no_grad():
            eye = torch.eye(self.hidden_dim)
            self.query_proj.weight.copy_(eye)
            self.key_proj.weight.copy_(eye)
            self.gate_s.weight.copy_(eye)
            self.gate_z.weight.copy_(eye)
            self.out_proj.weight.zero_()
            self.out_proj.weight[:, :self.hidden_dim].copy_(0.5 * eye)
            self.out_proj.weight[:, self.hidden_dim:].copy_(0.5 * eye)

    def forward(self, env_node_repr, prev_state=None, topk=None):
        # env_node_repr: [num_env_nodes, hidden_dim]
        num_nodes, hidden_dim = env_node_repr.shape
        if prev_state is None:
            prev_state = env_node_repr.mean(dim=0).detach()

        q = self.query_proj(prev_state)  # q_{t+1}
        keys = self.key_proj(env_node_repr)
        attn_logits = torch.matmul(keys, q) / np.sqrt(float(hidden_dim))

        if topk is None:
            topk = max(1, min(64, int(np.sqrt(num_nodes))))
        topk = min(topk, num_nodes)

        topk_scores, topk_indices = torch.topk(attn_logits, k=topk)
        topk_weights = F.softmax(topk_scores, dim=0)

        z_global = env_node_repr.mean(dim=0)
        z_sparse = torch.sum(topk_weights.unsqueeze(-1) * env_node_repr[topk_indices], dim=0)
        z_t = 0.5 * (z_global + z_sparse)

        lam = torch.sigmoid(self.gate_s(prev_state) + self.gate_z(z_t) + self.gate_b)
        s_t = lam * prev_state + (1.0 - lam) * z_t
        p_t = self.out_proj(torch.cat([z_t, s_t], dim=0))

        anchor_token = env_node_repr.new_zeros(num_nodes)
        key_readout = torch.sum(env_node_repr[topk_indices] * p_t.unsqueeze(0), dim=1)
        key_readout = torch.sigmoid(key_readout)
        anchor_token[topk_indices] = topk_weights * key_readout

        return anchor_token, s_t, p_t, topk_indices


def calculate_test_acc(graph, model, device):
    model.eval()
    graph = graph.to(device)

    y = graph.y
    out = model.inference(graph, "test")
    test_mask = graph.test_mask

    _, pred = out.max(dim=1)
    if test_mask.sum().item() > 0:
        correct = int(pred[test_mask].eq(y[test_mask]).sum().item())
        acc = correct / len(y[test_mask])
        return acc
    else:
        return 0.


def calculate_new_nodes_test_acc(graph, model, device):
    model.eval()
    graph = graph.to(device)

    y = graph.y
    out = model.inference(graph, "test")
    test_mask = graph.new_nodes_test_mask

    _, pred = out.max(dim=1)
    if test_mask.sum().item() > 0:
        correct = int(pred[test_mask].eq(y[test_mask]).sum().item())
        acc = correct / len(y[test_mask])
        return acc
    else:
        return 0.


def calculate_PM(acc_list):
    aa = sum(acc_list) / len(acc_list)
    return aa


def calculate_FM(final_acc_list, acc_list):
    tmp_list = [final_acc_list[i] - acc_list[i] for i in range(len(acc_list))]
    if len(tmp_list) <= 1:
        return 0.0
    return sum(tmp_list) / (len(tmp_list) - 1)


def evaluate(t, model_path, graph, device):
    model = torch.load(os.path.join(model_path, f'{t}.pt'))
    model = model.to(device)
    model.eval()

    y = graph.y
    out = model.inference(graph, "test")
    test_mask = graph.test_mask

    _, pred = out.max(dim=1)
    if test_mask.sum().item() > 0:
        correct = int(pred[test_mask].eq(y[test_mask]).sum().item())
        acc = correct / len(y[test_mask])
        return acc
    else:
        return 0.


def evaluate_Knowledge_feedback(args, time_step, device):
    dataset = DyDataset(dataset_name=args.dataset_name, edge_type=args.edge_type, m_size=args.m_size)
    data, _, _, _ = dataset.construct_knowledge_feed_graph(time_step)

    data = data.to(device)

    model_path = os.path.join(args.save_models_path, args.dataset_name)
    max_time_step = len(dataset) - 1
    task_acc = evaluate(max_time_step, model_path, data, device)
    return task_acc


def calculate_test_cluster(args, time_step, device):
    dataset = DyDataset(dataset_name=args.dataset_name, edge_type=args.edge_type, m_size=args.m_size)
    data, _, _, labels_num = dataset.construct_knowledge_feed_graph(time_step)
    data = data.to(device)

    model_path = os.path.join(args.save_models_path, args.dataset_name)
    max_time_step = len(dataset) - 1

    model = torch.load(os.path.join(model_path, f'{max_time_step}.pt'))
    model = model.to(device)
    model.eval()

    out = model.inference(data, "test", "cluster")
    test_mask = data.test_mask
    y = data.y
    y = y[test_mask]
    out = out[test_mask]
    np_y = y.detach().cpu().numpy()

    unique_categories = sorted(set(np_y))
    mapping = {old: new for new, old in enumerate(unique_categories)}
    y_new = [mapping[x] for x in np_y]
    n_clusters = np.unique(y_new).shape[0]

    np_embedding = out.detach().cpu().numpy()
    y_pred = KMeans(
        n_clusters=n_clusters,
        random_state=2025,
        max_iter=500,
        algorithm='auto'
    ).fit_predict(np_embedding)
    # visualization(np_embedding, np_y, 'cluster')
    # np_y_one_hot = np.zeros((len(np_y), labels_num), dtype=int)
    # np_y_one_hot[np.arange(len(np_y)), np_y] = 1
    #
    # y_pred_one_hot = np.zeros((len(np_y), labels_num), dtype=int)
    # y_pred_one_hot[np.arange(len(np_y)), y_pred] = 1

    NMI = normalized_mutual_info_score(y_new, y_pred)
    ARI = adjusted_rand_score(y_new, y_pred)
    return NMI, ARI


def calculate_test_non_community_detection(args, time_step, device):
    dataset = DyDataset(dataset_name=args.dataset_name, edge_type=args.edge_type, m_size=args.m_size)
    data, _, _, labels_num = dataset.construct_knowledge_feed_graph(time_step)
    data = data.to(device)
    y = data.y
    np_y = y.detach().cpu().numpy()
    model_path = os.path.join(args.save_models_path, args.dataset_name)
    max_time_step = len(dataset) - 1

    model = torch.load(os.path.join(model_path, f'{max_time_step}.pt'))
    model = model.to(device)
    model.eval()
    out = model.inference(data, "test", "cluster")
    np_embedding = out.detach().cpu().numpy()
    NMI, ARI = non_overlapping_main(np_embedding, np_y)
    return NMI, ARI


def shannon_entropy(matrix):
    matrix = F.softmax(matrix, dim=1).clamp_min(1e-12)
    log_matrix = torch.log2(matrix)
    elementwise_product = matrix * log_matrix
    entropy = -torch.sum(elementwise_product, dim=1)
    # avg_entropy = torch.mean(entropy)

    return entropy.squeeze()


def kl_divergence(x, y):
    x_log = F.log_softmax(x, dim=1)
    y = F.softmax(y, dim=1)
    kl = nn.KLDivLoss(reduction='none')
    out = kl(x_log, y)
    out = torch.sum(out, dim=1)
    return out


def rec_graph_structure_key(data):
    """Stable fingerprint from subgraph topology (feature/label agnostic)."""
    n = int(data.x.size(0))
    ei = data.edge_index.detach().cpu()
    lo = torch.minimum(ei[0], ei[1])
    hi = torch.maximum(ei[0], ei[1])
    pairs = torch.stack([lo, hi], dim=1)
    pairs_np = pairs.numpy()
    order = np.lexsort((pairs_np[:, 1], pairs_np[:, 0]))
    pairs = torch.from_numpy(pairs_np[order]).contiguous()
    h = hashlib.blake2b(digest_size=16)
    h.update(str(n).encode())
    h.update(pairs.numpy().tobytes())
    return h.hexdigest()


def _state_dict_shapes_equal(sd_a, sd_b):
    if set(sd_a.keys()) != set(sd_b.keys()):
        return False
    for k in sd_a.keys():
        if sd_a[k].shape != sd_b[k].shape:
            return False
    return True


def main(args, model, dataset, device, num_lambda):
    model = model.to(device)
    
    acc_list = []
    final_acc_list = []
    average_time_list = []
    history_task_dataloaders_list = []
    model_list = []
    post_snapshot_state_dicts = []
    rec_env_structure_keys = []
    node_embeddings_list = []
    selective_modeling_adjs = []
    anchor_token_list = []
    anchor_state = None
    anchor_module = None

    for id in range(len(dataset)):
        (rec_graph, ret_graph, now_graph) = dataset[
            id]
        print(f'********************************task id is {id}********************************')

        if rec_graph is not None:
            rec_graph = rec_graph.to(device)
        if ret_graph is not None:
            ret_graph = ret_graph.to(device)
        if now_graph is not None:
            now_graph = now_graph.to(device)

        if id == 0:
            print("-" * 10 + "in init epochs" + "-" * 10)
            training_time = 0
            for epoch in range(args.init_epochs):
                loss, time, data_loader_init, rec_graph_embeddings = model.init_train(
                    rec_graph,
                    phase='init',
                    args=args,
                    device=device
                )
                training_time += time
                # if epoch % 10 == 0:
                #     print(f"epoch:\t{epoch} and loss:\t{loss}")


            _, prev_snap_data, _, _, _ = generate_causal_labels(args, 1, dataset, device=device)
            prev_snap_data = prev_snap_data.to(device)
            node_embeddings = model(prev_snap_data.x, prev_snap_data.edge_index)  # torch.Size([2708, 7])
            node_embeddings_list.append(node_embeddings)

            # env_repr = model.inference(rec_graph, phase='init', task_class="cluster")
            env_repr = node_embeddings
            if anchor_module is None:
                anchor_module = SparseEnvAnchorMemory(hidden_dim=env_repr.shape[-1]).to(device)
            anchor_token, anchor_state, _, _ = anchor_module(env_repr, prev_state=anchor_state)
            anchor_token_list.append(anchor_token)

            acc = calculate_test_acc(rec_graph, model, device)
            print('Accuracy: {:.4f}'.format(acc))

            acc_list.append(acc)
            average_time_list.append(training_time / max(args.init_epochs, 1))
            save_model_path = os.path.join(args.save_models_path, args.dataset_name)
            os.makedirs(save_model_path, exist_ok=True)
            torch.save(model, os.path.join(save_model_path, f'{id}.pt'))
            history_task_dataloaders_list.append(data_loader_init)
            model_list.append(copy.deepcopy(model).to(device))
            post_snapshot_state_dicts.append(copy.deepcopy(model.state_dict()))

        else:
            old_inner_weights = copy.deepcopy(model.state_dict())

            print("-" * 20 + "Knowledge correction stage (inner loop)" + "-" * 20)
            model.open_parameters()
            rectify_time = 0
            for rectify_epoch in range(args.rectify_epochs):
                rectify_loss, tim, data_loader_rectify, rec_graph_embeddings_2 = model.rectify(
                    rectify_graph=rec_graph,
                    phase='rectify',
                    epoch=rectify_epoch,
                    args=args,
                    device=device
                )
                rectify_time += tim
                if rectify_loss == 0:
                    break
            history_task_dataloaders_list.append(data_loader_rectify)

            _, prev_snap_data, _, A_prev_dense, A_curr_dense = generate_causal_labels(
                args, int(id), dataset, device=device
            )
            prev_snap_data = prev_snap_data.to(device)
            node_embeddings = model(prev_snap_data.x, prev_snap_data.edge_index, phase='mamba')
            node_embeddings_list.append(node_embeddings)
            if int(id) - 1 == 0:
                selective_modeling_adjs.append(A_prev_dense.to(device))
            selective_modeling_adjs.append(A_curr_dense.to(device))

            # env_repr = model.inference(rec_graph, phase='rectify', task_class="cluster")
            env_repr = node_embeddings
            if anchor_module is None:
                anchor_module = SparseEnvAnchorMemory(hidden_dim=env_repr.shape[-1]).to(device)
            anchor_token, anchor_state, _, _ = anchor_module(env_repr, prev_state=anchor_state)
            anchor_token_list.append(anchor_token)

            curr_rec_structure_key = rec_graph_structure_key(rec_graph)
            skip_param_expand = False

            if args.ParameterE is True:
                print("-" * 20 + "Parameter expansion enabled" + "-" * 20)
                reused_snap = None
                for hist_idx in range(len(rec_env_structure_keys) - 2, -1, -1):
                    if rec_env_structure_keys[hist_idx] != curr_rec_structure_key:
                        continue
                    snap_sd = post_snapshot_state_dicts[hist_idx + 1]
                    if not _state_dict_shapes_equal(snap_sd, model.state_dict()):
                        continue
                    load_sd = {k: (v.to(device) if torch.is_tensor(v) else v) for k, v in snap_sd.items()}
                    model.load_state_dict(load_sd, strict=True)
                    reused_snap = hist_idx + 1
                    finetune_rounds = max(1, min(20, args.rectify_epochs // 20))
                    print(f"Detected repeated rec_graph structure from snapshot {reused_snap}; loaded that checkpoint and fine-tuned for {finetune_rounds} rounds without adding expansion parameters")
                    for ft in range(finetune_rounds):
                        fl, _, _, _ = model.rectify(
                            rectify_graph=rec_graph,
                            phase='rectify',
                            epoch=ft,
                            args=args,
                            device=device,
                        )
                        if fl == 0:
                            break
                    skip_param_expand = True
                    break

                if (
                    not skip_param_expand
                    and rec_env_structure_keys
                    and rec_env_structure_keys[-1] == curr_rec_structure_key
                ):
                    skip_param_expand = True
                    print("Current rec_graph matches previous snapshot structure; skipping parameter expansion and keeping corrected weights")

                if not skip_param_expand:
                    model.expand()
                    model.init_expanded_parameters()
                    model.isolate_parameters()
                    model.to(device)
                    model.combine()
                else:
                    model.to(device)
            else:
                print("-" * 20 + "Parameter expansion disabled" + "-" * 20)

            ########################################## MAMBA#################################################
            if args.MemoryAnchor is True:
                print("-" * 20 + "Memory anchor enabled" + "-" * 20)
                z_mp = torch.stack(node_embeddings_list)
                graph_ssm_input = z_mp.mean(dim=-1, keepdim=True)
                graph_ssm_input = graph_ssm_input.permute(2, 0, 1)  # 1, timestamps, nodes

                anchor_token = anchor_token_list[-1].unsqueeze(0).unsqueeze(0)  # 1, 1, nodes
                if anchor_token.size(-1) != graph_ssm_input.size(-1):
                    node_dim = graph_ssm_input.size(-1)
                    if anchor_token.size(-1) > node_dim:
                        anchor_token = anchor_token[..., :node_dim]
                    else:
                        pad = node_dim - anchor_token.size(-1)
                        anchor_token = F.pad(anchor_token, (0, pad))
                graph_ssm_input = torch.cat([anchor_token, graph_ssm_input], dim=1)  # 1, timestamps+1, nodes

                node_dim = graph_ssm_input.size(-1)
                anchor_adj = torch.eye(node_dim, device=device)
                selective_modeling_adjs_with_anchor = [anchor_adj] + selective_modeling_adjs

                DGSM = Mamba(d_model=node_dim,  # Model dimension d_model
                            d_state=16,  # SSM state expansion factor
                            d_conv=4,  # Local convolution width
                            expand=1,
                            num_snapshots=graph_ssm_input.size(1),
                            use_fast_path=False).to(device)
                LayerNorm = nn.LayerNorm(node_dim, eps=1e-10).to(device)
                graph_ssm_output = DGSM(graph_ssm_input, selective_modeling_adjs_with_anchor)  # 1, timestamps+1, nodes

                graph_ssm_output = LayerNorm(graph_ssm_output)
                # Calculate kl_loss and entropy on (num_snapshots+1, num_nodes)
                graph_ssm_output_2d = graph_ssm_output.squeeze(0)
                graph_ssm_input_2d = graph_ssm_input.squeeze(0)
                kl_loss = kl_divergence(graph_ssm_output_2d, graph_ssm_input_2d)
                entropy = shannon_entropy(graph_ssm_output_2d)
                inter_loss = entropy + 0.1 * kl_loss
                inter_loss = torch.mean(inter_loss).item()
                print(f"DG-Mamba loss with memory anchor: {inter_loss}")
            else:
                print("-" * 20 + "Memory anchor disabled" + "-" * 20)
                z_mp = torch.stack(node_embeddings_list)
                graph_ssm_input = z_mp.mean(dim=-1, keepdim=True)
                graph_ssm_input = graph_ssm_input.permute(2, 0, 1)  # 1, timestamps, nodes

                node_dim = graph_ssm_input.size(-1)
                selective_modeling_adjs_with_anchor = selective_modeling_adjs

                DGSM = Mamba(d_model=node_dim,  # Model dimension d_model
                            d_state=16,  # SSM state expansion factor
                            d_conv=4,  # Local convolution width
                            expand=1,
                            num_snapshots=graph_ssm_input.size(1),
                            use_fast_path=False).to(device)
                LayerNorm = nn.LayerNorm(node_dim, eps=1e-10).to(device)
                graph_ssm_output = DGSM(graph_ssm_input, selective_modeling_adjs_with_anchor)  # 1, timestamps+1, nodes

                graph_ssm_output = LayerNorm(graph_ssm_output)
                # Calculate kl_loss and entropy on (num_snapshots+1, num_nodes)
                graph_ssm_output_2d = graph_ssm_output.squeeze(0)
                graph_ssm_input_2d = graph_ssm_input.squeeze(0)
                kl_loss = kl_divergence(graph_ssm_output_2d, graph_ssm_input_2d)
                entropy = shannon_entropy(graph_ssm_output_2d)
                inter_loss = entropy + 0.1 * kl_loss
                inter_loss = torch.mean(inter_loss).item()
                print(f"DG-Mamba loss without memory anchor: {inter_loss}")

            if args.BackKF is True:
                print("-" * 20 + "Knowledge backpropagation stage (outer loop) enabled" + "-" * 20)
                old_inner_weights = copy.deepcopy(model.state_dict())
                bt_model = BackwardTransferDynamicGNN(model, lr_inner=0.01, lr_outer=0.001, lambda_meta=num_lambda)
                bt_model = bt_model.to(device)
                double_ring_loss = bt_model.learn_new_task(
                    loss=rectify_loss + inter_loss,
                    history_task_dataloaders=history_task_dataloaders_list,
                    old_inner_weights=old_inner_weights,
                    model_list=model_list,
                    device=device,
                    anchor_module=anchor_module,
                )
                print(f"Outer-loop knowledge backpropagation loss: {double_ring_loss}")
            else:
                print("-" * 20 + "Knowledge backpropagation stage (outer loop) disabled" + "-" * 20)
                model.to(device)

            # retrain_time = 0
            # for retrain_epoch in range(args.retrain_epochs):
            #     loss, time = model.retrain(retrain_graph=ret_graph, phase='retrain', epoch=retrain_epoch, args=args, device=device)
            #     retrain_time += time
            #     if loss == 0:
            #         break
            # average_retrain_time = retrain_time / args.retrain_epochs
            # model.combine()

            acc = calculate_test_acc(now_graph, model, device)
            print('Accuracy: {:.4f}'.format(acc))

            acc_list.append(acc)
            # average_time_list.append((rectify_time + retrain_time) / (args.rectify_epochs + args.retrain_epochs))
            average_time_list.append(rectify_time / max(args.rectify_epochs, 1))
            save_model_path = os.path.join(args.save_models_path, args.dataset_name)
            os.makedirs(save_model_path, exist_ok=True)
            torch.save(model, os.path.join(save_model_path, f'{id}.pt'))
            rec_env_structure_keys.append(curr_rec_structure_key)
            post_snapshot_state_dicts.append(copy.deepcopy(model.state_dict()))

    for id in range(len(dataset)):
        (rec_graph, ret_graph, now_graph) = dataset[id]

        if rec_graph is not None:
            rec_graph = rec_graph.to(device)
        if now_graph is not None:
            now_graph = now_graph.to(device)

        if id == 0:
            final_acc = calculate_test_acc(rec_graph, model, device)
            final_acc_list.append(final_acc)
        else:
            final_acc = calculate_test_acc(now_graph, model, device)
            final_acc_list.append(final_acc)

    PM = calculate_PM(acc_list)
    FM = calculate_FM(final_acc_list, acc_list)
    train_avg_time = sum(average_time_list) / len(average_time_list)
    return PM, FM, acc_list, train_avg_time


if __name__ == '__main__':
    time_start = time.time()
    args = get_args()
    device = get_device(args.cuda_enable)
    set_random_seed(args.seed)

    dataset = DyDataset(dataset_name=args.dataset_name, edge_type=args.edge_type,
                        m_size=args.m_size)  # memory_size=args.m_size
    in_dim = dataset.in_dim
    out_dim = dataset.labels_num
    model = GraphSage(in_dim=in_dim, out_dim=out_dim, multi_threading=args.multi_threading, normalize=args.normalize)

    model = model.to(device)

    num_lambda = args.lambda_meta
    PM, FM, acc_list, train_avg_time = main(
        args, model, dataset, device, num_lambda
    )
    print(f'average training time is: {train_avg_time}')
    print(f"PM is: {np.round(PM, 4)}, FM is: {np.round(FM, 4)}, acc is: {acc_list}")

    print("****************************************Node classification****************************************")
    task_acc_list = []
    for time_step in range(len(dataset)):
        task_acc = evaluate_Knowledge_feedback(args, time_step, device)
        task_acc_list.append(task_acc)
        print(f"Snapshot {time_step} ACC: {np.round(task_acc, 4)}")
    print(f"Node classification average ACC across snapshots: {np.round(np.mean(task_acc_list), 4)}")
    # time_end = time.time()
    # final_time = (time_end - time_start) / 60

    print("****************************************Node clustering****************************************")
    cluster_nmi_list = []
    cluster_ari_list = []
    for time_step in range(len(dataset)):
        nmi, ari = calculate_test_cluster(args, time_step, device)
        cluster_nmi_list.append(nmi)
        cluster_ari_list.append(ari)
        print(f"Snapshot {time_step} NMI: {np.round(nmi, 4)}")
        print(f"Snapshot {time_step} ARI: {np.round(ari, 4)}")
    print(f"Average NMI across all snapshots: {np.round(np.mean(cluster_nmi_list), 4)}")
    print(f"Average ARI across all snapshots: {np.round(np.mean(cluster_ari_list), 4)}")

    print("****************************************Non-overlapping community detection****************************************")
    ncd_nmi_list = []
    ncd_ari_list = []
    for time_step in range(len(dataset)):
        ncd_nmi, ncd_ari = calculate_test_non_community_detection(args, time_step, device)
        ncd_nmi_list.append(ncd_nmi)
        ncd_ari_list.append(ncd_ari)
        print(f"Snapshot {time_step} NMI: {np.round(ncd_nmi, 4)}")
        print(f"Snapshot {time_step} ARI: {np.round(ncd_ari, 4)}")
    print(f"Average NMI across all snapshots: {np.round(np.mean(ncd_nmi_list), 4)}")
    print(f"Average ARI across all snapshots: {np.round(np.mean(ncd_ari_list), 4)}")
    time_end = time.time()
    final_time = (time_end - time_start) / 60
    print(f"Model runtime: {final_time} min")
