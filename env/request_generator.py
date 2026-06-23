from __future__ import annotations

import numpy as np

from env.container import Container


class RequestGenerator:
    def __init__(self, config: dict, catalog: list[Container], seed: int = 42):
        self.config = config
        self.catalog = catalog
        self.num_nodes = config["num_nodes"]
        self.num_types = len(catalog)
        self.alpha = config["zipf_alpha"]
        self.timestep = 0
        self._seed = seed

        self._id_by_rank = {c.popularity_rank: c.id for c in catalog}
        self._initial_id_by_rank = dict(self._id_by_rank)

        ranks = np.arange(self.num_types)
        weights = 1.0 / np.power(ranks + 1, self.alpha)
        self._probabilities = weights / weights.sum()

        self.rng = np.random.default_rng(seed)

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

        container_ids = [self._id_by_rank[rank] for rank in range(self.num_types)]
        self.rng.shuffle(container_ids)
        self._id_by_rank = {
            rank: container_ids[rank] for rank in range(self.num_types)
        }

    def _maybe_burst(self) -> int | None:
        """With burst_probability, pick a container to receive burst_multiplier demand."""
        if self._traffic_pattern() != "bursty":
            return None

        burst_probability = float(self.config.get("burst_probability", 0.05))
        if self.rng.random() >= burst_probability:
            return None

        rank = int(self.rng.integers(0, self.num_types))
        return self._id_by_rank[rank]

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
                requests.append(self._id_by_rank[rank])
        return requests

    def peek(self) -> list[int | None]:
        """Return the next request batch without advancing RNG, timestep, or popularity."""
        rng_state = self.rng.bit_generator.state
        id_by_rank = dict(self._id_by_rank)

        self._maybe_shift_popularity()
        burst_container = self._maybe_burst()
        requests = self._sample_requests(burst_container)

        self.rng.bit_generator.state = rng_state
        self._id_by_rank = id_by_rank
        return requests

    def generate(self) -> list[int | None]:
        self._maybe_shift_popularity()
        burst_container = self._maybe_burst()
        requests = self._sample_requests(burst_container)
        self.timestep += 1
        return requests

    def reset(self) -> None:
        self.timestep = 0
        self._id_by_rank = dict(self._initial_id_by_rank)
        self.rng = np.random.default_rng(self._seed)
