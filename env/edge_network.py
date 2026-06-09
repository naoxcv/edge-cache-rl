from __future__ import annotations

import numpy as np

from env.edge_node import EdgeNode


class EdgeNetwork:
    def __init__(self, config: dict):
        self.config = config
        self.num_nodes = config["num_nodes"]
        self.num_clusters = config["num_clusters"]
        self.nodes = [EdgeNode(i, config["cache_capacity"]) for i in range(self.num_nodes)]
        self.cluster_for_node = {
            node_id: node_id % self.num_clusters for node_id in range(self.num_nodes)
        }
        self.adjacency = self._build_adjacency()
        self.latency_matrix = self._compute_latency_matrix(config)
        self.cloud_latency = config["cloud_latency_ms"]

    def _cluster_nodes(self, cluster_id: int) -> list[int]:
        return [
            node_id
            for node_id in range(self.num_nodes)
            if self.cluster_for_node[node_id] == cluster_id
        ]

    def _add_edge(self, u: int, v: int) -> None:
        if v not in self.adjacency[u]:
            self.adjacency[u].append(v)
        if u not in self.adjacency[v]:
            self.adjacency[v].append(u)

    def _build_adjacency(self) -> dict[int, list[int]]:
        adjacency = {node_id: [] for node_id in range(self.num_nodes)}

        for cluster_id in range(self.num_clusters):
            members = self._cluster_nodes(cluster_id)
            for i, u in enumerate(members):
                for v in members[i + 1 :]:
                    if v not in adjacency[u]:
                        adjacency[u].append(v)
                    if u not in adjacency[v]:
                        adjacency[v].append(u)

        for cluster_id in range(self.num_clusters - 1):
            left = self._cluster_nodes(cluster_id)[0]
            right = self._cluster_nodes(cluster_id + 1)[0]
            if right not in adjacency[left]:
                adjacency[left].append(right)
            if left not in adjacency[right]:
                adjacency[right].append(left)

        for node_id in adjacency:
            adjacency[node_id].sort()

        return adjacency

    def _compute_latency_matrix(self, config: dict) -> np.ndarray:
        intra = config["intra_cluster_latency_ms"]
        inter = config["inter_cluster_latency_ms"]
        matrix = np.zeros((self.num_nodes, self.num_nodes), dtype=np.float32)

        for i in range(self.num_nodes):
            for j in range(i + 1, self.num_nodes):
                if self.cluster_for_node[i] == self.cluster_for_node[j]:
                    matrix[i, j] = intra
                    matrix[j, i] = intra
                else:
                    matrix[i, j] = inter
                    matrix[j, i] = inter

        return matrix

    def get_neighbors(self, node_id: int) -> list[int]:
        return list(self.adjacency[node_id])

    def find_any_neighbor_with(self, node_id: int, container_id: int) -> int | None:
        for neighbor in self.adjacency[node_id]:
            if self.nodes[neighbor].is_cached(container_id):
                return neighbor
        return None

    def get_forwarding_cost(self, from_node: int, to_node: int) -> float:
        return float(self.latency_matrix[from_node, to_node])

    def reset(self) -> None:
        for node in self.nodes:
            node.reset()
