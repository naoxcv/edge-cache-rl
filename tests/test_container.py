import pytest

from env.container import create_catalog


def test_create_catalog_count():
    k = 20
    catalog = create_catalog(k)
    assert len(catalog) == k


def test_create_catalog_unique_ids():
    catalog = create_catalog(20)
    ids = [c.id for c in catalog]
    assert len(ids) == len(set(ids))


def test_create_catalog_ranks_cover_zero_to_k_minus_one():
    k = 20
    catalog = create_catalog(k)
    ranks = {c.popularity_rank for c in catalog}
    assert ranks == set(range(k))


def test_create_catalog_sorted_by_popularity_rank():
    catalog = create_catalog(10)
    ranks = [c.popularity_rank for c in catalog]
    assert ranks == list(range(10))


def test_create_catalog_deterministic_seeding():
    assert create_catalog(20, seed=42) == create_catalog(20, seed=42)
    assert create_catalog(20, seed=0) != create_catalog(20, seed=42)


def test_create_catalog_invalid_num_types():
    with pytest.raises(ValueError, match="num_types must be >= 1"):
        create_catalog(0)
