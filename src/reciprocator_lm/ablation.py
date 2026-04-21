import math
from functools import lru_cache
from typing import Optional


def _normalize_mode_sizes(mode_sizes: tuple[int, ...], state_rank: int) -> tuple[int, ...]:
    if len(mode_sizes) != state_rank:
        raise ValueError("mode size tuple length must match state_rank")
    normalized = tuple(sorted(int(size) for size in mode_sizes))
    if any(size <= 0 for size in normalized):
        raise ValueError("mode sizes must contain positive integers")
    return normalized


@lru_cache(maxsize=None)
def _divisors(value: int) -> tuple[int, ...]:
    if value <= 0:
        raise ValueError("capacity must be positive")
    divisors = set()
    limit = int(math.isqrt(value))
    for factor in range(1, limit + 1):
        if value % factor != 0:
            continue
        divisors.add(factor)
        divisors.add(value // factor)
    return tuple(sorted(divisors))


@lru_cache(maxsize=None)
def factor_tuples(capacity: int, state_rank: int, min_factor: int = 1) -> tuple[tuple[int, ...], ...]:
    """Return non-decreasing integer tuples whose product is the requested capacity."""

    if capacity <= 0:
        raise ValueError("capacity must be positive")
    if state_rank <= 0:
        raise ValueError("state_rank must be positive")

    if state_rank == 1:
        if capacity < min_factor:
            return ()
        return ((capacity,),)

    tuples: list[tuple[int, ...]] = []
    for factor in _divisors(capacity):
        if factor < min_factor:
            continue
        remainder = capacity // factor
        if factor * remainder != capacity:
            continue
        for suffix in factor_tuples(remainder, state_rank - 1, factor):
            tuples.append((factor, *suffix))
    return tuple(tuples)


def _spread_cost(mode_sizes: tuple[int, ...]) -> float:
    if len(mode_sizes) <= 1:
        return 0.0
    log_sizes = [math.log(size) for size in mode_sizes]
    mean_log = sum(log_sizes) / len(log_sizes)
    return sum((value - mean_log) ** 2 for value in log_sizes)


def _pair_cost(init_mode_sizes: tuple[int, ...], max_mode_sizes: tuple[int, ...]) -> tuple[float, ...]:
    growth_logs = tuple(math.log(max_size / init_size) for init_size, max_size in zip(init_mode_sizes, max_mode_sizes))
    mean_growth = sum(growth_logs) / len(growth_logs)
    growth_spread = sum((value - mean_growth) ** 2 for value in growth_logs)
    # Prefer balanced asymmetric factorizations when several exact-capacity
    # tuples are feasible so modes do not default to interchangeable cubes.
    return (
        -len(set(max_mode_sizes)),
        -len(set(init_mode_sizes)),
        _spread_cost(max_mode_sizes),
        _spread_cost(init_mode_sizes),
        growth_spread,
        sum(growth_logs),
        *max_mode_sizes,
        *init_mode_sizes,
    )


def select_mode_size_pair(
    *,
    state_rank: int,
    init_mode_sizes: Optional[tuple[int, ...]] = None,
    max_mode_sizes: Optional[tuple[int, ...]] = None,
    init_capacity: Optional[int] = None,
    max_capacity: Optional[int] = None,
) -> tuple[tuple[int, ...], tuple[int, ...]]:
    """Resolve init/max mode sizes with exact capacities and init <= max elementwise."""

    if state_rank <= 0:
        raise ValueError("state_rank must be positive")
    if init_capacity is not None and init_capacity <= 0:
        raise ValueError("init_capacity must be positive when provided")
    if max_capacity is not None and max_capacity <= 0:
        raise ValueError("max_capacity must be positive when provided")

    init_candidates = (
        (_normalize_mode_sizes(init_mode_sizes, state_rank),)
        if init_mode_sizes is not None
        else factor_tuples(int(init_capacity), state_rank)
        if init_capacity is not None
        else ()
    )
    max_candidates = (
        (_normalize_mode_sizes(max_mode_sizes, state_rank),)
        if max_mode_sizes is not None
        else factor_tuples(int(max_capacity), state_rank)
        if max_capacity is not None
        else ()
    )

    if not init_candidates and not max_candidates:
        raise ValueError("at least one of mode sizes or capacities must be provided")
    if not init_candidates:
        init_candidates = max_candidates
    if not max_candidates:
        max_candidates = init_candidates

    feasible_pairs = [
        (init_sizes, max_sizes)
        for init_sizes in init_candidates
        for max_sizes in max_candidates
        if all(init_size <= max_size for init_size, max_size in zip(init_sizes, max_sizes))
    ]
    if not feasible_pairs:
        raise ValueError(
            "no feasible mode-size pair satisfies the requested capacities with init_mode_sizes <= max_mode_sizes"
        )

    best_init, best_max = min(feasible_pairs, key=lambda pair: _pair_cost(pair[0], pair[1]))
    return best_init, best_max


__all__ = ["factor_tuples", "select_mode_size_pair"]
