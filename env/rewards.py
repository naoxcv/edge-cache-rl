from __future__ import annotations

from typing import Any

from env.edge_network import EdgeNetwork


def score_cache_request(
    network: EdgeNetwork,
    node_id: int,
    container_id: int | None,
    config: dict[str, Any],
    *,
    enable_forwarding: bool | None = None,
    same_cluster_only: bool | None = None,
) -> float:
    """Score one request: local hit, optional forward hit, or cloud pull.

    Updates the node's hit / forward / miss counters and LRU touch on local hits.
    Shared by CachingEnv and MultiAgentCachingEnv so forwarding flags stay aligned.
    """
    if container_id is None:
        return 0.0

    if enable_forwarding is None:
        enable_forwarding = bool(config.get("enable_forwarding", True))
    if same_cluster_only is None:
        same_cluster_only = bool(config.get("forwarding_same_cluster_only", True))

    node = network.nodes[node_id]
    if node.is_cached(container_id):
        node.touch_container(container_id)
        node.hits += 1
        return float(config["reward_local_hit"])

    if enable_forwarding:
        neighbor = network.find_any_neighbor_with(
            node_id,
            container_id,
            same_cluster_only=same_cluster_only,
        )
        if neighbor is not None:
            node.forwards += 1
            return float(config["reward_forward_hit"])

    node.misses += 1
    return float(config["reward_cloud_pull"])
