import torch
import torch.nn as nn
import torch.nn.functional as F
from torch_geometric.nn import GCNConv
from torch_geometric.data import Data
import numpy as np
import copy


def generate_causal_labels(data_prev: Data, data_curr: Data) -> torch.Tensor:
    """
    Generate the causal label matrix Y_t^{(c)} from consecutive snapshots.

    Args:
        data_prev (Data): Graph snapshot at time t-1.
        data_curr (Data): Graph snapshot at time t.

    Returns:
        torch.Tensor: Causal label matrix Y_t^{(c)} with shape [N, N].
    """
    A_prev = data_prev.edge_index
    A_curr = data_curr.edge_index

    num_nodes = data_prev.num_nodes

    A_prev_dense = torch.zeros(num_nodes, num_nodes, dtype=torch.float, device=A_prev.device)
    A_curr_dense = torch.zeros(num_nodes, num_nodes, dtype=torch.float, device=A_prev.device)

    if A_prev.shape[1] > 0:
        A_prev_dense[A_prev[0], A_prev[1]] = 1
    if A_curr.shape[1] > 0:
        A_curr_dense[A_curr[0], A_curr[1]] = 1

    delta_A = A_curr_dense - A_prev_dense
    delta_X = data_curr.x - data_prev.x

    edge_cause_mask = (delta_A != 0)

    attr_change_indicator = (delta_X.sum(dim=1) != 0)
    attr_cause_mask = attr_change_indicator.repeat(num_nodes, 1)

    potential_cause_mask = edge_cause_mask | attr_cause_mask

    Y_causal = potential_cause_mask * A_prev_dense

    return Y_causal


class CGDRL(nn.Module):
    """
    English documentation line.
    """

    def __init__(self, input_dim: int, hidden_dim: int, output_dim: int, num_layers: int = 2):
        super(CGDRL, self).__init__()
        self.convs = nn.ModuleList()
        self.convs.append(GCNConv(input_dim, hidden_dim))
        for _ in range(num_layers - 2):
            self.convs.append(GCNConv(hidden_dim, hidden_dim))
        if num_layers > 1:
            self.convs.append(GCNConv(hidden_dim, hidden_dim))

        self.classifier = nn.Linear(hidden_dim, output_dim)
        self.dropout = nn.Dropout(0.5)

    def encode(self, x: torch.Tensor, edge_index: torch.Tensor) -> torch.Tensor:
        """GCN encoder."""
        for i, conv in enumerate(self.convs):
            x = conv(x, edge_index)
            if i < len(self.convs) - 1:
                x = F.relu(x)
                x = self.dropout(x)
        return x

    def forward(self, data_curr: Data, h_prev: torch.Tensor) -> tuple:
        """
        English documentation line.

        Args:
            data_curr (Data): Current graph snapshot at time t.
            h_prev (torch.Tensor): Node representations from time t-1.

        Returns:
            tuple: (current representation h_t, main-task logits, causal score matrix S_t)
        """
        h_t = self.encode(data_curr.x, data_curr.edge_index)

        main_logits = self.classifier(h_t)

        # h_t: [N, d], h_prev: [N, d] -> S_t: [N, N]
        causal_scores = torch.sigmoid(torch.matmul(h_t, h_prev.t()))

        return h_t, main_logits, causal_scores


def generate_dynamic_data(num_nodes: int, num_features: int, num_snapshots: int, num_classes: int) -> list:
    """Generate synthetic dynamic graph data."""
    snapshots = []

    A_0 = torch.randint(0, 2, (2, num_nodes * 5))
    A_0 = A_0.unique(dim=1)
    X_0 = torch.randint(0, 2, (num_nodes, num_features), dtype=torch.float)
    y_0 = torch.randint(0, num_classes, (num_nodes,))
    train_mask = torch.zeros(num_nodes, dtype=torch.bool)
    train_mask[:int(0.8 * num_nodes)] = True
    data_0 = Data(x=X_0, edge_index=A_0, y=y_0, train_mask=train_mask)
    snapshots.append(data_0)

    for t in range(1, num_snapshots):
        prev_data = snapshots[-1]
        A_prev_dense = torch.zeros(num_nodes, num_nodes)
        if prev_data.edge_index.shape[1] > 0:
            A_prev_dense[prev_data.edge_index[0], prev_data.edge_index[1]] = 1

        A_curr_dense = copy.deepcopy(A_prev_dense)
        X_curr = copy.deepcopy(prev_data.x)

        num_edge_changes = max(1, int(0.05 * num_nodes))
        for _ in range(num_edge_changes):
            i, j = np.random.randint(0, num_nodes, 2)
            A_curr_dense[i, j] = 1 - A_curr_dense[i, j]

        num_attr_changes = max(1, int(0.1 * num_nodes))
        attr_nodes = np.random.choice(num_nodes, num_attr_changes, replace=False)
        for node in attr_nodes:
            feat_to_change = np.random.randint(0, num_features)
            X_curr[node, feat_to_change] = 1 - X_curr[node, feat_to_change]

        edge_index_curr = A_curr_dense.nonzero().t().contiguous()

        y_curr = torch.randint(0, num_classes, (num_nodes,))

        data_curr = Data(x=X_curr, edge_index=edge_index_curr, y=y_curr, train_mask=train_mask)
        snapshots.append(data_curr)

    return snapshots


if __name__ == '__main__':
    NUM_NODES = 50
    NUM_FEATURES = 16
    NUM_SNAPSHOTS = 10
    NUM_CLASSES = 3
    HIDDEN_DIM = 32
    OUTPUT_DIM = NUM_CLASSES
    NUM_LAYERS = 2
    LAMBDA_CAUSAL = 0.5
    LEARNING_RATE = 0.01
    EPOCHS_PER_SNAPSHOT = 50  //

    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    print(f"Using device: {device}")

    snapshots = generate_dynamic_data(NUM_NODES, NUM_FEATURES, NUM_SNAPSHOTS, NUM_CLASSES)
    for i, data in enumerate(snapshots):
        snapshots[i] = data.to(device)

    model = CGDRL(NUM_FEATURES, HIDDEN_DIM, OUTPUT_DIM, NUM_LAYERS).to(device)
    optimizer = torch.optim.Adam(model.parameters(), lr=LEARNING_RATE, weight_decay=5e-4)
    criterion_main = nn.CrossEntropyLoss()
    criterion_causal = nn.BCELoss()

    print("Start training...")
    h_prev = None

    for t in range(1, NUM_SNAPSHOTS):
        print(f"\n--- Training time step: {t} ---")
        data_prev = snapshots[t - 1]
        data_curr = snapshots[t]

        Y_causal = generate_causal_labels(data_prev, data_curr).to(device)

        if h_prev is None:
            with torch.no_grad():
                h_prev = model.encode(data_prev.x, data_prev.edge_index)

        model.train()
        for epoch in range(EPOCHS_PER_SNAPSHOT):
            h_t, main_logits, causal_scores = model(data_curr, h_prev)

            loss_main = criterion_main(main_logits[data_curr.train_mask], data_curr.y[data_curr.train_mask])

            loss_causal = criterion_causal(causal_scores, Y_causal)

            total_loss = loss_main + LAMBDA_CAUSAL * loss_causal

            optimizer.zero_grad()
            total_loss.backward()
            optimizer.step()

            if (epoch + 1) % 10 == 0:
                print(f"  Epoch {epoch + 1}/{EPOCHS_PER_SNAPSHOT}, Total Loss: {total_loss.item():.4f}, "
                      f"Main Loss: {loss_main.item():.4f}, Causal Loss: {loss_causal.item():.4f}")

        h_prev = h_t.detach()

    print("\nTraining complete!")

    print("\nEvaluate on the final snapshot...")
    model.eval()
    with torch.no_grad():
        final_data = snapshots[-1]
        final_h, final_logits, _ = model(final_data, h_prev)
        pred = final_logits.argmax(dim=1)
        correct = (pred[final_data.train_mask] == final_data.y[final_data.train_mask]).sum()
        acc = int(correct) / int(final_data.train_mask.sum())
        print(f"Final training-set accuracy: {acc:.4f}")

