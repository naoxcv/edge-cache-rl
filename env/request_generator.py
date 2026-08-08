from __future__ import annotations

import copy
from collections.abc import Mapping

import numpy as np

from env.container import Container


class RequestGenerator:
    """Zipf-weighted, locality-aware request stream with optional shifting/bursty traffic."""

    def __init__(
        self,
        config: dict,
        catalog: list[Container],
        seed: int = 42,
        cluster_for_node: Mapping[int, int] | None = None,
    ) -> None:
        self.config = config
        self.catalog = catalog
        self.num_nodes = config["num_nodes"]
        self.num_types = len(catalog)
        self.alpha = config["zipf_alpha"]
        self.locality_factor = float(config.get("locality_factor", 0.0))
        if not 0.0 <= self.locality_factor <= 1.0:
            raise ValueError(
                f"locality_factor must be in [0, 1]; got {self.locality_factor}"
            )
        self.timestep = 0
        self._seed = seed

        num_clusters = int(config.get("num_clusters", 1))
        if cluster_for_node is None:
            self.cluster_for_node = {
                node_id: node_id % num_clusters for node_id in range(self.num_nodes)
            }
        else:
            self.cluster_for_node = {
                node_id: int(cluster_for_node[node_id])
                for node_id in range(self.num_nodes)
            }

        ranks = np.arange(self.num_types)
        weights = 1.0 / np.power(ranks + 1, self.alpha)
        self._probabilities = weights / weights.sum()

        self.rng = np.random.default_rng(seed)
        self._global_ids_by_rank = [
            c.id for c in sorted(catalog, key=lambda c: c.popularity_rank)
        ]
        self._cluster_rank_permutations, self._node_rank_permutations = (
            self._build_rank_permutations()
        )
        self._id_by_rank_per_node = self._materialize_node_rankings()
        self._initial_global_ids_by_rank = list(self._global_ids_by_rank)
        self._initial_id_by_rank_per_node = [
            dict(mapping) for mapping in self._id_by_rank_per_node
        ]
        self._initial_rng_state = copy.deepcopy(self.rng.bit_generator.state)
        # Compatibility alias: node 0's local popularity ranking.
        self._id_by_rank = self._id_by_rank_per_node[0]

    def _swap_permutation(self, num_swaps: int) -> np.ndarray:
        permutation = np.arange(self.num_types)
        for _ in range(num_swaps):
            left, right = self.rng.choice(self.num_types, size=2, replace=False)
            permutation[left], permutation[right] = (
                permutation[right],
                permutation[left],
            )
        return permutation

    def _build_rank_permutations(
        self,
    ) -> tuple[dict[int, np.ndarray], list[np.ndarray]]:
        """Create hierarchical rank permutations.

        ``locality_factor=0`` gives every node the global ranking. At 1, every
        node is independently shuffled. Intermediate values use cluster-level
        swaps plus fewer node-level swaps, so nodes in the same cluster remain
        more similar than nodes in different clusters.
        """
        clusters = sorted(set(self.cluster_for_node.values()))
        if self.locality_factor == 0.0:
            identity = np.arange(self.num_types)
            return (
                {cluster: identity.copy() for cluster in clusters},
                [identity.copy() for _ in range(self.num_nodes)],
            )

        if self.locality_factor == 1.0:
            cluster_permutations = {
                cluster: self.rng.permutation(self.num_types) for cluster in clusters
            }
            node_permutations = [
                self.rng.permutation(self.num_types) for _ in range(self.num_nodes)
            ]
            return cluster_permutations, node_permutations

        cluster_swaps = int(round(self.locality_factor * self.num_types))
        node_swaps = int(round(self.locality_factor * self.num_types / 3.0))
        cluster_permutations = {
            cluster: self._swap_permutation(cluster_swaps) for cluster in clusters
        }
        node_permutations = [
            self._swap_permutation(node_swaps) for _ in range(self.num_nodes)
        ]
        return cluster_permutations, node_permutations

    def _materialize_node_rankings(self) -> list[dict[int, int]]:
        rankings: list[dict[int, int]] = []
        for node_id in range(self.num_nodes):
            cluster_perm = self._cluster_rank_permutations[
                self.cluster_for_node[node_id]
            ]
            node_perm = self._node_rank_permutations[node_id]
            rank_indices = cluster_perm[node_perm]
            rankings.append(
                {
                    rank: self._global_ids_by_rank[int(global_rank)]
                    for rank, global_rank in enumerate(rank_indices)
                }
            )
        return rankings

    def popularity_ranking(self, node_id: int) -> tuple[int, ...]:
        """Container ids from hottest to coldest for ``node_id``."""
        mapping = self._id_by_rank_per_node[node_id]
        return tuple(mapping[rank] for rank in range(self.num_types))

    def _traffic_pattern(self) -> str:
        return str(self.config.get("traffic_pattern", "stationary"))

    def _maybe_shift_popularity(self) -> None:
        """Permute rank assignments so former hot containers become cold and vice versa."""
        if self._traffic_pattern() != "shifting":
            return

        shift_interval = int(self.config.get("shift_interval", 500))
        if shift_interval <= 0:
            return
        if self.timestep <= 0 or self.timestep % shift_interval != 0:
            return

        self.rng.shuffle(self._global_ids_by_rank)
        self._id_by_rank_per_node = self._materialize_node_rankings()
        self._id_by_rank = self._id_by_rank_per_node[0]

    def _maybe_burst(self) -> int | None:
        """With burst_probability, pick a container to receive burst_multiplier demand."""
        if self._traffic_pattern() != "bursty":
            return None

        burst_probability = float(self.config.get("burst_probability", 0.05))
        if self.rng.random() >= burst_probability:
            return None

        rank = int(self.rng.integers(0, self.num_types))
        return self._id_by_rank_per_node[0][rank]

    def _sample_requests(self, burst_container: int | None = None) -> list[int | None]:
        requests: list[int | None] = []
        burst_slots = 0
        if burst_container is not None:
            burst_slots = min(
                int(self.config.get("burst_multiplier", 10)),
                self.num_nodes,
            )

        for node_idx in range(self.num_nodes):
            if node_idx < burst_slots:
                requests.append(burst_container)
            else:
                rank = int(self.rng.choice(self.num_types, p=self._probabilities))
                requests.append(self._id_by_rank_per_node[node_idx][rank])
        return requests

    def peek(self) -> list[int | None]:
        """Return the next request batch without advancing RNG, timestep, or popularity."""
        rng_state = copy.deepcopy(self.rng.bit_generator.state)
        global_ids_by_rank = list(self._global_ids_by_rank)
        id_by_rank_per_node = [
            dict(mapping) for mapping in self._id_by_rank_per_node
        ]

        self._maybe_shift_popularity()
        burst_container = self._maybe_burst()
        requests = self._sample_requests(burst_container)

        self.rng.bit_generator.state = rng_state
        self._global_ids_by_rank = global_ids_by_rank
        self._id_by_rank_per_node = id_by_rank_per_node
        self._id_by_rank = self._id_by_rank_per_node[0]
        return requests

    def generate(self) -> list[int | None]:
        """Sample one request per node, advance timestep and traffic dynamics."""
        self._maybe_shift_popularity()
        burst_container = self._maybe_burst()
        requests = self._sample_requests(burst_container)
        self.timestep += 1
        return requests

    def reset(self) -> None:
        """Restore initial RNG state, rankings, and timestep to zero."""
        self.timestep = 0
        self._global_ids_by_rank = list(self._initial_global_ids_by_rank)
        self._id_by_rank_per_node = [
            dict(mapping) for mapping in self._initial_id_by_rank_per_node
        ]
        self._id_by_rank = self._id_by_rank_per_node[0]
        self.rng = np.random.default_rng(self._seed)
        self.rng.bit_generator.state = copy.deepcopy(self._initial_rng_state)
