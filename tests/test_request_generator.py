from collections import Counter

import pytest

from configs import load_config
from env.container import create_catalog
from env.request_generator import RequestGenerator


@pytest.fixture
def config():
    return load_config()


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


def test_maybe_burst_stub_returns_none(generator):
    assert generator._maybe_burst() is None


def test_maybe_shift_popularity_stub_is_noop(generator):
    original = dict(generator._id_by_rank)
    generator._maybe_shift_popularity()
    assert generator._id_by_rank == original
