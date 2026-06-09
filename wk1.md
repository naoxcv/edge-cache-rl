# Week 1 Backlog

## Setup
- [ ] Create project structure: `env/`, `agents/`, `experiments/`, `analysis/`, `tests/`, `configs/`
- [ ] Add `__init__.py` to each package
- [ ] Create `requirements.txt` (gymnasium, numpy, networkx, pyyaml, pytest, matplotlib)
- [ ] Create venv, install deps
- [ ] Create `.gitignore` (venv/, __pycache__/, *.pyc, .pytest_cache/)
- [ ] Write README with setup instructions

## Config (`configs/default.yaml`)
- [ ] Define all parameters: num_nodes (10), num_clusters (3), cache_capacity (5), num_container_types (20), zipf_alpha (1.0), episode_length (1000), observation_window (50)
- [ ] Include latency values: intra_cluster (2ms), inter_cluster (20ms), cloud (100ms)
- [ ] Include reward values: local_hit (1.0), forward_hit (0.5), cloud_pull (-1.0)
- [ ] Include traffic params: pattern (stationary), shift_interval (500), burst_probability (0.05), burst_multiplier (10)
- [ ] Stub future params: comm_level (0), comm_penalty_lambda (0.0), selective_comm_threshold (0.1), learning_rate (0.0003), buffer_size (100000), hidden_layers ([128,128,64])
- [ ] Write `load_config()` utility function

## Container Catalog (`env/container.py`)
- [ ] Container dataclass: `id`, `popularity_rank`
- [ ] `create_catalog(num_types) -> list[Container]` — returns containers ranked 0 to K-1 by popularity
- [ ] Deterministic seeding for reproducibility
- [ ] Tests: correct count, unique IDs, ranks cover 0 to K-1

## Request Generator (`env/request_generator.py`)
- [ ] `__init__` takes config and catalog, initializes Zipf distribution from popularity ranks
- [ ] `generate() -> list[int | None]` — returns one container ID per node sampled from Zipf
- [ ] `_maybe_shift_popularity()` — no-op stub for week 3
- [ ] `_maybe_burst() -> int | None` — no-op stub for week 3
- [ ] `reset()` — resets timestep and popularity ordering
- [ ] Deterministic seeding via `np.random.default_rng(seed=42)`
- [ ] Tests: distribution matches Zipf (top containers get most requests over 10k samples), all IDs valid, deterministic seeding produces identical sequences, reset reproduces same sequence

## Edge Node (`env/edge_node.py`)
- [ ] `__init__`: node_id, cache_capacity, cache (list, ordered by recency), cache_set (set for O(1) lookup), request_history (list), hit/miss/forward counters
- [ ] `is_cached(container_id) -> bool`
- [ ] `cache_container(container_id) -> int | None` — returns evicted ID if cache was full
- [ ] `evict_container(container_id) -> bool` — manual eviction for RL agent
- [ ] `record_request(container_id)` — append to history, maintain sliding window
- [ ] `get_state(num_container_types, observation_window) -> np.ndarray` — returns vector of shape (2K+1,): cache binary (K,) + utilization (1,) + request frequency (K,) normalized to [0,1]
- [ ] `reset()` — clear cache, history, counters
- [ ] Tests: cache/lookup works, eviction returns correct ID, full cache behavior, get_state returns shape (2K+1,), sliding window respected, reset clears everything

## Edge Network (`env/edge_network.py`)
- [ ] `__init__` takes config, creates N EdgeNodes, builds topology, computes latency matrix
- [ ] `_build_topology(config) -> nx.Graph` — hierarchical: nodes assigned round-robin to clusters, fully connected within cluster, one link between adjacent clusters
- [ ] `_compute_latency_matrix(config) -> np.ndarray` — N×N, intra-cluster or inter-cluster based on cluster membership, diagonal is zero
- [ ] `get_neighbors(node_id) -> list[int]`
- [ ] `find_nearest_cached(node_id, container_id) -> int | None` — lowest-latency neighbor with container cached, None if nobody has it
- [ ] `get_forwarding_cost(from_node, to_node) -> float`
- [ ] `reset()` — resets all nodes
- [ ] Stub mesh and star topologies (raise NotImplementedError for now)
- [ ] Tests: correct cluster count, latency matrix is symmetric, diagonal is zero, intra < inter latency, get_neighbors correct, find_nearest_cached returns None when appropriate, returns lowest-latency neighbor when multiple have it

## Gymnasium Environment (`env/caching_env.py`)
- [ ] Subclass `gymnasium.Env`
- [ ] `__init__`: create catalog, network, request generator from config
- [ ] Single-node mode for now (active_node = 0), multi-node comes week 4
- [ ] `observation_space`: Box(0, 1, shape=(2K+1,))
- [ ] `action_space`: Discrete(2K+1) — actions 0..K-1 cache container k, K..2K-1 evict container k, 2K is no-op
- [ ] `reset()` — reset network and request generator, return initial observation
- [ ] `step(action)` — execute action, generate request, compute reward (local hit → +1.0, forward hit → +0.5, cloud pull → -1.0, no request → 0.0), record request, advance timestep, return (obs, reward, terminated, truncated, info)
- [ ] `info` dict includes cache_hit_rate and timestep
- [ ] `render()` — print cache state and metrics for debugging
- [ ] Tests: reset returns correct shape, step with no-op returns valid output, cache then request → local hit reward, uncached with no neighbor → cloud pull reward, episode terminates at episode_length, cache_hit_rate in info is between 0 and 1

## Baselines (`agents/baselines.py`)
- [ ] `LRUPolicy.act(observation, requested) -> int` — if requested not cached and cache full, evict least recently used and cache it; if not full, cache it; if cached or no request, no-op
- [ ] `LFUPolicy.act(observation, requested) -> int` — same logic but evict least frequently used based on request history
- [ ] Both return action integers compatible with env action space
- [ ] Tests: LRU evicts oldest entry, LFU evicts least frequent entry

## Validation (`validate_week1.py`)
- [ ] Run LRU and LFU each for 10,000 timesteps
- [ ] Print cache hit rate, forward rate, cloud pull rate for both
- [ ] Sanity check: hit rates should be 15–50% for K=20, C=5 under Zipf
- [ ] Plot cumulative reward over time for both, save as `results/week1_baselines.png`
- [ ] If hit rate is 0% or 100%, something is broken

## Final
- [ ] All tests pass (`pytest tests/`)
- [ ] Validation script runs end-to-end with sane numbers
- [ ] README documents environment interface (obs space, action space, reward)
- [ ] README includes baseline results
- [ ] Clean commit, tag `v0.1-environment`