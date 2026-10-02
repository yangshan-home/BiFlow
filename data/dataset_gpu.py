import os
import torch
import random
import numpy as np

from torch_geometric.data import Data
from torch_geometric.utils import k_hop_subgraph
from torch.utils.data import Dataset
from torch_geometric.utils import to_undirected
from data.config import DATA_CONFIG


class DyDataset(Dataset):
    def __init__(self, dataset_name='dblp', edge_type='stream_edges', m_size=128):
        self.dataset_name = dataset_name
        self.edge_type = edge_type
        config = DATA_CONFIG[dataset_name]
        self.basic_t = config['basic_t']
        self.hop_num = config['hop_num']
        self.task_nums = config['task_nums']
        self.datapath = os.path.join('./data', dataset_name)
        self.m_size = m_size

        self.datalist = os.listdir(os.path.join(self.datapath, edge_type))
        self.features = torch.tensor(np.loadtxt(os.path.join(self.datapath, 'attributes')), dtype=torch.float)

        self.labeled_nodes = np.loadtxt(os.path.join(self.datapath, 'labels'))[:, 0]
        self.label_nodes_dict = {self.labeled_nodes[i]: i for i in range(len(self.labeled_nodes))}
        self.labels = torch.tensor(np.loadtxt(os.path.join(self.datapath, 'labels'))[:, 1], dtype=torch.long)

        self.train_nodes_set = set(np.loadtxt(os.path.join(self.datapath, 'train_nodes')))
        self.test_nodes_set = set(np.loadtxt(os.path.join(self.datapath, 'test_nodes')))
        self.val_nodes_set = set(np.loadtxt(os.path.join(self.datapath, 'val_nodes')))

        self.basic_edges, self.basic_nodes_set = self.load_basic_edges()
        self.labels_num = torch.max(self.labels) + 1
        self.in_dim = self.features.shape[1]
        self.memory = list()

    def __getitem__(self, index):
        if index == 0:
            edges = self.basic_edges
            edges = torch.tensor(edges.transpose(), dtype=torch.long)
            unique_nodes, inverse_indices = torch.unique(edges, return_inverse=True)
            reindexed_edges = inverse_indices.reshape(edges.shape)
            edge_index = reindexed_edges

            nodes = torch.tensor(list(self.basic_nodes_set), dtype=torch.long)
            sorted_nodes = sorted(nodes)
            features = self.features[sorted_nodes]

            label_idx = [self.label_nodes_dict[item.item()] if item.item() in self.label_nodes_dict else 0 for item in sorted_nodes]
            labels = self.labels[label_idx]

            train_set = self.basic_nodes_set.intersection(self.train_nodes_set)
            test_set = self.basic_nodes_set.intersection(self.test_nodes_set)
            val_set = self.basic_nodes_set.intersection(self.val_nodes_set)

            train_mask = np.array([sorted_nodes[i].item() in train_set for i in range(len(sorted_nodes))],
                                    dtype=bool)
            test_mask = np.array([sorted_nodes[i].item() in test_set for i in range(len(sorted_nodes))],
                                    dtype=bool)
            val_mask = np.array([sorted_nodes[i].item() in val_set for i in range(len(sorted_nodes))],
                                    dtype=bool)
            train_mask = torch.tensor(np.argwhere(train_mask).flatten().tolist(), dtype=torch.long)
            test_mask = torch.tensor(np.argwhere(test_mask).flatten().tolist(), dtype=torch.long)
            val_mask = torch.tensor(np.argwhere(val_mask).flatten().tolist(), dtype=torch.long)

            graph = Data(x=features, edge_index=edge_index, y=labels)
            graph.train_mask = train_mask
            graph.test_mask = test_mask
            graph.val_mask = val_mask
            graph.new_nodes_test_mask = test_mask
            self._update_memory(nodes=list(train_set))
            return graph, None, None
        else:
            previous_snap_edges = self.basic_edges
            for i in range(self.basic_t, index):
                previous_snap_edges = np.concatenate(
                    (previous_snap_edges, np.loadtxt(os.path.join(self.datapath, self.edge_type, str(i)))), axis=0)
            previous_all_snaps_nodes_set = set(previous_snap_edges.flatten().tolist())

            previous_snap_edges = torch.tensor(previous_snap_edges.transpose(), dtype=torch.long)
            current_snap_edges = np.loadtxt(os.path.join(self.datapath, self.edge_type, str(index)))
            current_snap_nodes_set = set(current_snap_edges.flatten().tolist())
            previous_current_snaps_intersection_nodes_set = previous_all_snaps_nodes_set.intersection(current_snap_nodes_set)
            previous_current_snaps_union_nodes_set = previous_all_snaps_nodes_set.union(current_snap_nodes_set)

            memory_nodes_set = set(self.memory)

            current_snap_edges = torch.tensor(current_snap_edges.transpose(), dtype=torch.long)
            previous_current_snaps_cat_edges = torch.cat((previous_snap_edges, current_snap_edges), dim=1)
            rec_memory_previous_current_snaps_intersection_nodes_set = memory_nodes_set.union(previous_current_snaps_intersection_nodes_set)

            unique_nodes, inverse_indices = torch.unique(previous_snap_edges, return_inverse=True)
            reindexed_edges = inverse_indices.reshape(previous_snap_edges.shape)

            previous_all_snaps_nodes_set_tensor = torch.tensor(list(set(previous_snap_edges.flatten().tolist())), dtype=torch.long)
            previous_all_snaps_nodes_set_tensor = sorted(previous_all_snaps_nodes_set_tensor)
            rec_x = self.features[previous_all_snaps_nodes_set_tensor]

            label_idx = [self.label_nodes_dict[item.item()] if item.item() in self.label_nodes_dict else 0 for item in previous_all_snaps_nodes_set_tensor]
            rec_y = self.labels[label_idx]
            memory_train_set = memory_nodes_set.intersection(self.train_nodes_set)
            added_nodes = current_snap_nodes_set - previous_all_snaps_nodes_set
            deleted_nodes = previous_all_snaps_nodes_set - current_snap_nodes_set
            changed_nodes_set = added_nodes.union(deleted_nodes)
            if len(changed_nodes_set) > 0:
                undirected_cat_edges = to_undirected(previous_current_snaps_cat_edges)
                affected_subset, _, _, _ = k_hop_subgraph(
                    node_idx=torch.tensor(list(changed_nodes_set), dtype=torch.long),
                    num_hops=self.hop_num,
                    edge_index=undirected_cat_edges,
                    relabel_nodes=False
                )
                last_affected_nodes_set = set(affected_subset.tolist())
            else:
                last_affected_nodes_set = previous_current_snaps_intersection_nodes_set
            last_affected_train_set = last_affected_nodes_set.intersection(self.train_nodes_set)
            memory_train_mask = np.array([previous_all_snaps_nodes_set_tensor[i].item() in memory_train_set for i in range(len(previous_all_snaps_nodes_set_tensor))], dtype=bool)
            last_affected_train_mask = np.array([previous_all_snaps_nodes_set_tensor[i].item() in last_affected_train_set for i in range(len(previous_all_snaps_nodes_set_tensor))], dtype=bool)

            memory_train_mask = torch.tensor(np.argwhere(memory_train_mask).flatten().tolist(), dtype=torch.long)
            last_affected_train_mask = torch.tensor(np.argwhere(last_affected_train_mask).flatten().tolist(), dtype=torch.long)

            rec_data = Data(x=rec_x, edge_index=reindexed_edges, y=rec_y)
            rec_data.memory_train_mask = memory_train_mask
            rec_data.last_affected_train_mask = last_affected_train_mask

            ###############################################################################################################
            ret_unique_nodes, ret_inverse_indices = torch.unique(previous_current_snaps_cat_edges, return_inverse=True)
            ret_reindexed_edges = ret_inverse_indices.reshape(previous_current_snaps_cat_edges.shape)
            ret_nodes_set = memory_nodes_set.union(current_snap_nodes_set)
            ret_basic_nodes_set = torch.tensor(list(set(previous_current_snaps_cat_edges.flatten().tolist())), dtype=torch.long)

            ret_edge_index = ret_reindexed_edges
            ret_basic_nodes_set = sorted(ret_basic_nodes_set)
            ret_x = self.features[ret_basic_nodes_set]

            label_idx = [self.label_nodes_dict[item.item()] if item.item() in self.label_nodes_dict else 0 for item in ret_basic_nodes_set]

            ret_y = self.labels[label_idx]

            affected_train_set = ret_nodes_set.intersection(self.train_nodes_set)
            affected_test_set = ret_nodes_set.intersection(self.test_nodes_set)
            affected_val_set = ret_nodes_set.intersection(self.val_nodes_set)

            memory_train_mask = np.array([ret_basic_nodes_set[i].item() in memory_train_set for i in range(len(ret_basic_nodes_set))], dtype=bool)

            affected_train_mask = np.array(
                [ret_basic_nodes_set[i].item() in affected_train_set for i in range(len(ret_basic_nodes_set))],
                dtype=bool)
            affected_test_mask = np.array([ret_basic_nodes_set[i].item() in affected_test_set for i in range(len(ret_basic_nodes_set))],
                                     dtype=bool)
            affected_val_mask = np.array([ret_basic_nodes_set[i].item() in affected_val_set for i in range(len(ret_basic_nodes_set))],
                                    dtype=bool)

            memory_train_mask = torch.tensor(np.argwhere(memory_train_mask).flatten().tolist(), dtype=torch.long)

            affected_train_mask = torch.tensor(np.argwhere(affected_train_mask).flatten().tolist(), dtype=torch.long)
            affected_test_mask = torch.tensor(np.argwhere(affected_test_mask).flatten().tolist(), dtype=torch.long)
            affected_val_mask = torch.tensor(np.argwhere(affected_val_mask).flatten().tolist(), dtype=torch.long)

            ret_data = Data(x=ret_x, edge_index=ret_edge_index, y=ret_y)
            ret_data.memory_train_mask = memory_train_mask
            ret_data.affected_train_mask = affected_train_mask
            ret_data.affected_test_mask = affected_test_mask
            ret_data.affected_val_mask = affected_val_mask

            ##############################################################################################################
            now_unique_nodes, now_inverse_indices = torch.unique(previous_current_snaps_cat_edges, return_inverse=True)
            now_reindexed_edges = now_inverse_indices.reshape(previous_current_snaps_cat_edges.shape)
            now_basic_nodes_set = torch.tensor(list(set(previous_current_snaps_cat_edges.flatten().tolist())),
                                               dtype=torch.long)
            now_edge_index = now_reindexed_edges
            now_basic_nodes_set = sorted(now_basic_nodes_set)
            now_x = self.features[now_basic_nodes_set]

            label_idx = [self.label_nodes_dict[item.item()] if item.item() in self.label_nodes_dict else 0 for item in now_basic_nodes_set]
            now_y = self.labels[label_idx]

            train_set = previous_current_snaps_union_nodes_set.intersection(self.train_nodes_set)
            test_set = previous_current_snaps_union_nodes_set.intersection(self.test_nodes_set)
            val_set = previous_current_snaps_union_nodes_set.intersection(self.val_nodes_set)

            new_nodes_test_set = current_snap_nodes_set.intersection(self.test_nodes_set)

            train_mask = torch.tensor([now_basic_nodes_set[i].item() in train_set for i in range(len(now_basic_nodes_set))],
                                      dtype=torch.bool)
            test_mask = torch.tensor([now_basic_nodes_set[i].item() in test_set for i in range(len(now_basic_nodes_set))],
                                     dtype=torch.bool)
            val_mask = torch.tensor([now_basic_nodes_set[i].item() in val_set for i in range(len(now_basic_nodes_set))],
                                    dtype=torch.bool)

            new_nodes_test_mask = np.array([now_basic_nodes_set[i].item() in new_nodes_test_set for i in range(len(now_basic_nodes_set))],
                                           dtype=bool)

            new_nodes_test_mask = torch.tensor(np.argwhere(new_nodes_test_mask).flatten().tolist(), dtype=torch.long)

            now_data = Data(x=now_x, edge_index=now_edge_index, y=now_y)
            now_data.train_mask = train_mask
            now_data.test_mask = test_mask
            now_data.val_mask = val_mask
            now_data.new_nodes_test_mask = new_nodes_test_mask
            self._update_memory(list(train_set))
            return rec_data, ret_data, now_data

    def construct_knowledge_feed_graph(self, index):
        edges_list = np.loadtxt(os.path.join(self.datapath, self.edge_type, str(index)))
        edges = torch.tensor(edges_list.transpose(), dtype=torch.long)
        nodes_set_ = set(edges.flatten().tolist())
        nodes_set = torch.tensor(list(nodes_set_), dtype=torch.long)
        sorted_nodes = sorted(nodes_set)
        node_indices = torch.tensor([node.item() for node in sorted_nodes])
        features = self.features[node_indices]
        label_idx = [self.label_nodes_dict[item.item()] if item.item() in self.label_nodes_dict else 0 for item in
                     sorted_nodes]
        labels = self.labels[label_idx]
        unique_nodes, inverse_indices = torch.unique(edges, return_inverse=True)
        reindexed_edges = inverse_indices.reshape(edges.shape)

        node_mapping = {original: new for new, original in enumerate(sorted_nodes)}
        test_mask = torch.tensor(list(node_mapping.values()), dtype=torch.long)
        # test_set = nodes_set_.intersection(int(node) for node in self.test_nodes_set)
        # test_mask = np.array([node.item() in test_set for node in sorted_nodes], dtype=bool)
        # test_mask = torch.tensor(np.argwhere(test_mask).flatten().tolist(), dtype=torch.long)

        graph = Data(x=features, edge_index=reindexed_edges, y=labels)
        graph.test_mask = test_mask
        return graph, None, None, torch.unique(self.labels).shape[0]

    def load_basic_edges(self):
        edges = None
        for i in range(self.basic_t):
            if i == 0:
                edges = np.loadtxt(os.path.join(self.datapath, self.edge_type, str(i)))
            else:
                edges = np.concatenate((edges, np.loadtxt(os.path.join(self.datapath, self.edge_type, str(i)))), axis=0)
        basic_nodes_set = set(edges.flatten().tolist())
        return edges, basic_nodes_set

    def _update_memory(self, nodes):
        # if len(nodes) <= self.m_size:
        #     self.memory = nodes
        # else:
        #     self.memory = random.sample(nodes, self.m_size)
        self.memory = nodes

    def __len__(self):
        return self.task_nums

