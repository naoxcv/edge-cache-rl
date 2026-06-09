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

    def _maybe_shift_popularity(self) -> None:
        pass

    def _maybe_burst(self) -> int | None:
        return None

    def _sample_requests(self) -> list[int | None]:
        requests: list[int | None] = []
        for _ in range(self.num_nodes):
            rank = int(self.rng.choice(self.num_types, p=self._probabilities))
            requests.append(self._id_by_rank[rank])
        return requests

    def peek(self) -> list[int | None]:
        """Return the next request batch without advancing RNG or timestep."""
        state = self.rng.bit_generator.state
        self._maybe_shift_popularity()
        self._maybe_burst()
        requests = self._sample_requests()
        self.rng.bit_generator.state = state
        return requests

    def generate(self) -> list[int | None]:
        self._maybe_shift_popularity()
        self._maybe_burst()
        requests = self._sample_requests()
        self.timestep += 1
        return requests

    def reset(self) -> None:
        self.timestep = 0
        self._id_by_rank = dict(self._initial_id_by_rank)
        self.rng = np.random.default_rng(self._seed)
