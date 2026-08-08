from collections import Counter

import pytest

from configs import load_config
from env.container import create_catalog
from env.request_generator import RequestGenerator


@pytest.fixture
def config():
    cfg = load_config()
    # Most generator unit tests assume a fixed Zipf ranking (no shifts).
    cfg["traffic_pattern"] = "stationary"
    cfg["locality_factor"] = 0.0
    return cfg


@pytest.fixture
def catalog(config):
    return create_catalog(config["num_container_types"], seed=42)


@pytest.fixture
def generator(config, catalog):
    return RequestGenerator(config, catalog, seed=42)


def test_generate_returns_one_request_per_node(generator, config):
    requests = generator.generate()

    assert len(requests) == config["num_nodes"]
    assert all(isinstance(r, int) for r in requests)


def test_generate_all_ids_valid(generator, catalog):
    valid_ids = {c.id for c in catalog}

    for _ in range(100):
        requests = generator.generate()
        assert all(r in valid_ids for r in requests)


def test_generate_zipf_distribution(config, catalog):
    gen = RequestGenerator(config, catalog, seed=42)
    id_by_rank = {c.popularity_rank: c.id for c in catalog}
    counts = Counter()

    for _ in range(10_000):
        counts[gen.generate()[0]] += 1

    most_popular = id_by_rank[0]
    least_popular = id_by_rank[config["num_container_types"] - 1]
    mid_rank = id_by_rank[config["num_container_types"] // 2]

    assert counts[most_popular] > counts[mid_rank]
    assert counts[mid_rank] > counts[least_popular]


def test_deterministic_seeding(config, catalog):
    gen_a = RequestGenerator(config, catalog, seed=42)
    gen_b = RequestGenerator(config, catalog, seed=42)

    seq_a = [gen_a.generate() for _ in range(10)]
    seq_b = [gen_b.generate() for _ in range(10)]

    assert seq_a == seq_b


def test_different_seeds_produce_different_sequences(config, catalog):
    gen_a = RequestGenerator(config, catalog, seed=42)
    gen_b = RequestGenerator(config, catalog, seed=7)

    seq_a = [gen_a.generate() for _ in range(10)]
    seq_b = [gen_b.generate() for _ in range(10)]

    assert seq_a != seq_b


def test_reset_reproduces_same_sequence(generator):
    before_reset = [generator.generate() for _ in range(10)]
    generator.reset()
    after_reset = [generator.generate() for _ in range(10)]

    assert before_reset == after_reset
    assert generator.timestep == 10


def test_reset_clears_timestep(generator):
    generator.generate()
    generator.generate()
    assert generator.timestep == 2

    generator.reset()
    assert generator.timestep == 0


def test_stationary_does_not_shift(generator):
    original = dict(generator._id_by_rank)
    generator.timestep = 500
    generator._maybe_shift_popularity()
    assert generator._id_by_rank == original


def test_shifting_permutes_rank_mapping(config, catalog):
    config = dict(config)
    config["traffic_pattern"] = "shifting"
    config["shift_interval"] = 100

    gen = RequestGenerator(config, catalog, seed=42)
    before = dict(gen._id_by_rank)

    gen.timestep = 100
    gen._maybe_shift_popularity()

    assert gen._id_by_rank != before
    assert sorted(gen._id_by_rank.keys()) == list(range(config["num_container_types"]))
    assert sorted(gen._id_by_rank.values()) == sorted(before.values())


def test_shifting_redistributes_demand(config, catalog):
    config = dict(config)
    config["traffic_pattern"] = "shifting"
    config["shift_interval"] = 500

    gen = RequestGenerator(config, catalog, seed=42)
    rank0_before_shift = gen._id_by_rank[0]

    before_counts = Counter(gen.generate()[0] for _ in range(500))
    after_counts = Counter(gen.generate()[0] for _ in range(500))

    assert before_counts[rank0_before_shift] > after_counts[rank0_before_shift]


def test_burst_inactive_for_stationary(generator):
    generator.config = dict(generator.config)
    generator.config["traffic_pattern"] = "stationary"
    assert generator._maybe_burst() is None


def test_burst_spikes_demand_for_one_container(config, catalog):
    config = dict(config)
    config["traffic_pattern"] = "bursty"
    config["burst_probability"] = 1.0
    config["burst_multiplier"] = 10
    config["num_nodes"] = 10

    gen = RequestGenerator(config, catalog, seed=42)
    requests = gen.generate()

    counts = Counter(requests)
    assert sum(counts.values()) == 10
    assert max(counts.values()) == config["burst_multiplier"]


def test_burst_probability_zero_never_bursts(config, catalog):
    config = dict(config)
    config["traffic_pattern"] = "bursty"
    config["burst_probability"] = 0.0

    gen = RequestGenerator(config, catalog, seed=42)
    for _ in range(100):
        assert gen._maybe_burst() is None


def test_peek_does_not_advance_timestep_or_shift(generator):
    generator.config = dict(generator.config)
    generator.config["traffic_pattern"] = "shifting"
    generator.config["shift_interval"] = 1

    generator.timestep = 1
    mapping_before = dict(generator._id_by_rank)
    peeked = generator.peek()

    assert generator.timestep == 1
    assert generator._id_by_rank == mapping_before
    assert len(peeked) == generator.num_nodes


def test_peek_matches_generate_when_no_mutation(generator):
    expected = generator.generate()
    generator.reset()
    assert generator.peek() == expected


def test_locality_zero_gives_identical_node_rankings(config, catalog):
    config = dict(config)
    config["locality_factor"] = 0.0
    gen = RequestGenerator(config, catalog, seed=42)

    rankings = [gen.popularity_ranking(node) for node in range(gen.num_nodes)]
    assert all(ranking == rankings[0] for ranking in rankings[1:])


def test_locality_one_gives_independent_node_rankings(config, catalog):
    config = dict(config)
    config["locality_factor"] = 1.0
    gen = RequestGenerator(config, catalog, seed=42)

    rankings = [gen.popularity_ranking(node) for node in range(gen.num_nodes)]
    assert len(set(rankings)) == gen.num_nodes


def _rank_distance(left: tuple[int, ...], right: tuple[int, ...]) -> float:
    right_position = {container_id: rank for rank, container_id in enumerate(right)}
    return sum(
        abs(rank - right_position[container_id])
        for rank, container_id in enumerate(left)
    ) / len(left)


@pytest.mark.parametrize("seed", [0, 7, 42])
def test_locality_point_three_is_more_similar_within_clusters(config, seed):
    config = dict(config)
    config["locality_factor"] = 0.3
    config["num_nodes"] = 12
    config["num_clusters"] = 3
    catalog = create_catalog(config["num_container_types"], seed=seed)
    gen = RequestGenerator(config, catalog, seed=seed)

    within: list[float] = []
    across: list[float] = []
    for left in range(gen.num_nodes):
        for right in range(left + 1, gen.num_nodes):
            distance = _rank_distance(
                gen.popularity_ranking(left),
                gen.popularity_ranking(right),
            )
            if gen.cluster_for_node[left] == gen.cluster_for_node[right]:
                within.append(distance)
            else:
                across.append(distance)

    assert sum(within) / len(within) < sum(across) / len(across)


def test_heterogeneous_reset_reproduces_requests_and_rankings(config, catalog):
    config = dict(config)
    config["locality_factor"] = 0.3
    gen = RequestGenerator(config, catalog, seed=42)
    rankings_before = [
        gen.popularity_ranking(node) for node in range(gen.num_nodes)
    ]
    requests_before = [gen.generate() for _ in range(10)]

    gen.reset()

    assert [
        gen.popularity_ranking(node) for node in range(gen.num_nodes)
    ] == rankings_before
    assert [gen.generate() for _ in range(10)] == requests_before


@pytest.mark.parametrize("factor", [-0.1, 1.1])
def test_invalid_locality_factor_raises(config, catalog, factor):
    config = dict(config)
    config["locality_factor"] = factor
    with pytest.raises(ValueError, match="locality_factor"):
        RequestGenerator(config, catalog, seed=42)
