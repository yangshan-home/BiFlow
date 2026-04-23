import torch
import torch.nn as nn
from torch_geometric.data import Data
import numpy as np
import copy
import os


def _numpy_edges_to_rc(arr):
    """Return row/col index tensors from np.loadtxt edge array, shape (E,2) or (2,E)."""
    ep = np.atleast_2d(np.asarray(arr, dtype=np.int64))
    if ep.size == 0:
        return np.array([], dtype=np.int64), np.array([], dtype=np.int64)
    if ep.shape[1] == 2:
        return ep[:, 0], ep[:, 1]
    if ep.shape[0] == 2:
        return ep[0], ep[1]
    raise ValueError(f"Unexpected edge array shape: {ep.shape}")


def generate_causal_labels(args, id, dataset, device=None):
    """
    Generate the causal label matrix Y_t^{(c)} from consecutive snapshots.

    English documentation line.
    """
    path_prev = os.path.join(dataset.datapath, dataset.edge_type, str(id - 1))
    path_curr = os.path.join(dataset.datapath, dataset.edge_type, str(id))
    reverse_prev_edges = np.loadtxt(path_prev)
    reverse_curr_edges = np.loadtxt(path_curr)

    n_nodes = int(dataset.features.shape[0])

    features = dataset.features
    if not isinstance(features, torch.Tensor):
        features = torch.tensor(features, dtype=torch.float32)
    if device is not None:
        features = features.to(device)

    dev = device if device is not None else features.device

    A_prev_dense = torch.zeros(n_nodes, n_nodes, dtype=torch.float32, device=dev)
    A_curr_dense = torch.zeros(n_nodes, n_nodes, dtype=torch.float32, device=dev)

    rp0, rp1 = _numpy_edges_to_rc(reverse_prev_edges)
    rc0, rc1 = _numpy_edges_to_rc(reverse_curr_edges)

    rp0 = torch.as_tensor(rp0, dtype=torch.long, device=dev)
    rp1 = torch.as_tensor(rp1, dtype=torch.long, device=dev)
    rc0 = torch.as_tensor(rc0, dtype=torch.long, device=dev)
    rc1 = torch.as_tensor(rc1, dtype=torch.long, device=dev)

    if rp0.numel():
        A_prev_dense[rp0, rp1] = 1
    if rc0.numel():
        A_curr_dense[rc0, rc1] = 1

    data_curr = features
    data_prev = features
    delta_A = A_curr_dense - A_prev_dense
    delta_X = data_curr - data_prev

    edge_cause_mask = delta_A != 0
    attr_change_indicator = delta_X.sum(dim=1) != 0
    attr_cause_mask = attr_change_indicator.unsqueeze(0).expand(n_nodes, -1)

    potential_cause_mask = edge_cause_mask | attr_cause_mask
    Y_causal = potential_cause_mask * A_prev_dense

    reverse_prev_edges_t = torch.stack([rp0, rp1], dim=0)
    reverse_curr_edges_t = torch.stack([rc0, rc1], dim=0)

    prev_snap_data = Data(x=data_prev, edge_index=reverse_prev_edges_t)
    curr_snap_data = Data(x=data_curr, edge_index=reverse_curr_edges_t)
    return Y_causal, prev_snap_data, curr_snap_data, A_prev_dense, A_curr_dense


class CGDRL(nn.Module):
    """
    English documentation line.
    """

    def __init__(self, curr_model):
        super(CGDRL, self).__init__()
        self.curr_model = curr_model
        self.optimizer = torch.optim.Adam(self.curr_model.parameters())

    def calculate_causal_loss(self, model_list, old_inner_weights, prev_snap_data, curr_snap_data, Y_causal,
                              device=None):
        criterion_causal = nn.BCELoss()

        if device is not None:
            criterion_causal = criterion_causal.to(device)

        if "expand_w" not in old_inner_weights:
            old_model = model_list[0]
            old_model.eval()
            if device is not None:
                old_model = old_model.to(device)
            h_prev = old_model(prev_snap_data.x, prev_snap_data.edge_index)
        else:
            old_model = copy.deepcopy(self.curr_model)
            old_model.load_state_dict(old_inner_weights)
            old_model.eval()
            if device is not None:
                old_model = old_model.to(device)
            h_prev = old_model(prev_snap_data.x, prev_snap_data.edge_index)

        self.curr_model.train()
        if device is not None:
            self.curr_model = self.curr_model.to(device)
        h_curr = self.curr_model(curr_snap_data.x, curr_snap_data.edge_index)
        causal_scores = torch.sigmoid(torch.matmul(h_curr, h_prev.t()))

        loss_causal = criterion_causal(causal_scores, Y_causal)
        loss_causal = 0.01 * loss_causal
        self.optimizer.zero_grad()
        loss_causal.backward()
        self.optimizer.step()
        h_prev = h_curr.detach()
        return h_prev, loss_causal
