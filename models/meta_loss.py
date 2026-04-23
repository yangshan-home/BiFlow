
import torch
import torch.nn.functional as F
import copy


class BackwardTransferDynamicGNN(torch.nn.Module):
    def __init__(
        self,
        model,
        lr_inner=0.01,
        lr_outer=0.001,
        lambda_meta=1,
        lambda_cons=0.001,
        w_relax=0.2,
    ):
        """
        Args:
            model: Backbone graph neural network (e.g., GCN, GAT)
            lr_inner: Inner-loop learning rate (fast adaptation)
            lr_outer: Outer-loop learning rate (meta optimization)
            lambda_meta: Weight for meta loss term (lambda)
            lambda_cons: Weight for representation consistency loss; defaults to `lambda_meta`.
            w_relax: Relaxation factor (0~1) for consistency weight when historical-task performance drops,
                     w_i=1 if performance improves, otherwise w_i=w_relax, to avoid consistently zero loss_cons.
        """
        super().__init__()
        self.model = model
        self.lr_inner = lr_inner
        self.lr_outer = lr_outer
        self.lambda_meta = lambda_meta
        self.lambda_cons = lambda_meta if lambda_cons is None else lambda_cons
        self.w_relax = w_relax
        self.batch_num = 4
        self.optimizer_outer = torch.optim.Adam(self.model.parameters(), lr=lr_outer)

        self.history_task_dataloaders = []

    @staticmethod
    def _move_adjs_to_device(adjs, device):
        """
        NeighborSampler adjs are usually list[(edge_index, e_id, size), ...].
        Only move edge_index to device to avoid calling .to() directly on tuples.
        """
        if not isinstance(adjs, list):
            return adjs.to(device) if hasattr(adjs, "to") else adjs

        moved = []
        for adj in adjs:
            if isinstance(adj, (tuple, list)) and len(adj) >= 1 and hasattr(adj[0], "to"):
                edge_index = adj[0].to(device)
                if len(adj) == 3:
                    moved.append((edge_index, adj[1], adj[2]))
                else:
                    rest = []
                    for k in range(1, len(adj)):
                        item = adj[k]
                        rest.append(item.to(device) if hasattr(item, "to") else item)
                    moved.append((edge_index, *rest))
            else:
                moved.append(adj.to(device) if hasattr(adj, "to") else adj)
        return moved

    @staticmethod
    def _env_repr_phase(model, anchor_phase: str) -> str:
        """
        Keep consistent with the batch path: `GraphSage.forward(..., phase)` defaults to 'retrain'.
        English documentation line.
        English documentation line.
        English documentation line.
        """
        try:
            conv0 = model.convs[0]
            if getattr(conv0, "expand_l_weights", None) is not None:
                return "retrain"
        except Exception:
            pass
        return anchor_phase

    def learn_new_task(
        self,
        loss,
        history_task_dataloaders,
        old_inner_weights,
        model_list,
        device,
        anchor_module=None,
        anchor_phase: str = "mamba",
        meta_epochs: int = 20,
    ):
        """
        English documentation line.

        Args:
            loss: Scalar or task-related auxiliary loss (treated as constant when gradient-free).
            history_task_dataloaders: Historical task data.
            old_inner_weights: Previous-task model state_dict.
            model_list: List of historical models.
            device: Device.
            anchor_module: Sparse environment anchor module.
            anchor_phase: Forward phase.
            meta_epochs: Number of outer-loop iterations (default 100).
        """
        if isinstance(loss, torch.Tensor):
            loss_scalar = float(loss.detach().cpu().item())
        else:
            loss_scalar = float(loss)

        last_total = torch.as_tensor(loss_scalar, device=device, dtype=torch.float32)

        if len(history_task_dataloaders) <= 1:
            return last_total

        hist_loaders = history_task_dataloaders[:-1]

        if "expand_w" not in old_inner_weights:
            old_model = model_list[0]
            old_model.eval()
            old_model = old_model.to(device)
        else:
            old_model = copy.deepcopy(self.model)
            old_model.load_state_dict(old_inner_weights)
            old_model.eval()
            old_model = old_model.to(device)

        phase_env = self._env_repr_phase(old_model, anchor_phase)

        baseline_losses = []
        for _, old_task_dataloaders in enumerate(hist_loaders):
            graph, old_task_dataloader = old_task_dataloaders
            x = graph.x.to(device)
            y = graph.y.to(device)
            with torch.no_grad():
                old_loss_sum = 0.0
                for old_data in old_task_dataloader:
                    batch_size, n_id, adjs = old_data
                    adjs = self._move_adjs_to_device(adjs, device)
                    n_id = n_id.to(device)
                    out_old = old_model(x[n_id], adjs)
                    old_loss_sum += F.cross_entropy(out_old, y[n_id[:batch_size]])
                baseline_old_loss = old_loss_sum / max(len(old_task_dataloader), 1)
            baseline_losses.append(baseline_old_loss)

        anchor_s_old_list = None
        anchor_p_old_list = None
        if anchor_module is not None and self.lambda_cons != 0:
            anchor_module.eval()
            for p in anchor_module.parameters():
                p.requires_grad_(False)

            anchor_s_old_list = []
            anchor_p_old_list = []
            prev_state = None
            with torch.no_grad():
                for old_task_dataloaders in hist_loaders:
                    graph, _ = old_task_dataloaders
                    graph = graph.to(device)
                    env_repr_old = old_model(
                        graph.x, graph.edge_index, phase=phase_env
                    )
                    _, prev_state, anchor_p_t, _ = anchor_module(env_repr_old, prev_state=prev_state)
                    anchor_s_old_list.append(prev_state)
                    anchor_p_old_list.append(anchor_p_t)

        for epoch in range(meta_epochs):
            self.optimizer_outer.zero_grad()

            loss_meta = torch.zeros((), device=device, dtype=torch.float32)
            loss_cons = torch.zeros((), device=device, dtype=torch.float32)

            for i, old_task_dataloaders in enumerate(hist_loaders):
                graph, old_task_dataloader = old_task_dataloaders
                x = graph.x.to(device)
                y = graph.y.to(device)
                baseline_old_loss = baseline_losses[i]

                current_loss_sum = 0.0
                for old_data in old_task_dataloader:
                    self.model.train()
                    batch_size, n_id, adjs = old_data
                    adjs = self._move_adjs_to_device(adjs, device)
                    n_id = n_id.to(device)
                    out_current = self.model(x[n_id], adjs)
                    current_loss_sum += F.cross_entropy(out_current, y[n_id[:batch_size]])
                denom = max(len(old_task_dataloader), 1)
                current_old_loss = current_loss_sum / denom

                meta_term = F.relu(current_old_loss - baseline_old_loss)
                loss_meta = loss_meta + meta_term

                better = (current_old_loss <= baseline_old_loss).float().detach()
                w_i = better + (1.0 - better) * self.w_relax
                if anchor_p_old_list is not None and i < len(anchor_p_old_list):
                    graph_full = graph.to(device)
                    phase_env_new = self._env_repr_phase(self.model, anchor_phase)
                    env_repr_new = self.model(
                        graph_full.x, graph_full.edge_index, phase=phase_env_new
                    )
                    prev_state_input = anchor_s_old_list[i - 1] if i > 0 else None
                    _, _, anchor_p_new, _ = anchor_module(
                        env_repr_new,
                        prev_state=prev_state_input,
                    )

                    anchor_diff = anchor_p_new - anchor_p_old_list[i]
                    loss_cons = loss_cons + w_i * anchor_diff.pow(2).mean()

            total_loss = (
                torch.as_tensor(loss_scalar, device=device, dtype=torch.float32)
                + self.lambda_meta * loss_meta
                + self.lambda_cons * loss_cons
            )

            if total_loss.requires_grad:
                total_loss.backward()
                self.optimizer_outer.step()

            last_total = total_loss.detach()

            if epoch == 0 or epoch == meta_epochs - 1 or (epoch + 1) % 20 == 0:
                lm = (self.lambda_meta * loss_meta).detach()
                lc = (self.lambda_cons * loss_cons).detach()
                print(
                    f"[learn_new_task] epoch {epoch + 1}/{meta_epochs} "
                    f"total={last_total.item():.6f} "
                    f"lambda_meta*meta={lm.item():.6f} lambda_cons*cons={lc.item():.6f}"
                )

        return last_total
