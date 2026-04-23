

import numpy as np
from sklearn.metrics import normalized_mutual_info_score, adjusted_rand_score
import community as community_louvain
from sklearn.neighbors import NearestNeighbors


def non_overlapping_main(embeddings, true_labels):
    k = 10
    nn = NearestNeighbors(n_neighbors=k, metric='cosine').fit(embeddings)
    distances, indices = nn.kneighbors(embeddings)

    import networkx as nx
    G = nx.Graph()
    for i in range(len(embeddings)):
        G.add_node(i)
    for i, neighbors in enumerate(indices):
        for j in neighbors:
            if i != j:
                # G.add_edge(i, j, weight=1.0 / (distances[i, list(neighbors).index(j)] + 1e-5))
                G.add_edge(i, j, weight=1.0)

    partition = community_louvain.best_partition(G)

    pred_labels = [partition[i] for i in range(len(embeddings))]

    nmi = normalized_mutual_info_score(true_labels, pred_labels)
    ari = adjusted_rand_score(true_labels, pred_labels)

    return nmi, ari