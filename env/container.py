import random
from dataclasses import dataclass


@dataclass(frozen=True)
class Container:
    id: int
    popularity_rank: int  # 0 = most popular (highest Zipf weight)


def create_catalog(num_types: int, seed: int = 42) -> list[Container]:
    """Build K containers with unique IDs and popularity ranks 0..K-1.

    Ranks are assigned via a seeded shuffle so runs are reproducible but
    container id != popularity rank in general. Returned list is sorted by
    popularity_rank (most popular first).
    """
    if num_types < 1:
        raise ValueError(f"num_types must be >= 1, got {num_types}")

    rng = random.Random(seed)
    rank_by_id = list(range(num_types))
    rng.shuffle(rank_by_id)

    catalog = [
        Container(id=i, popularity_rank=rank_by_id[i])
        for i in range(num_types)
    ]
    return sorted(catalog, key=lambda c: c.popularity_rank)
