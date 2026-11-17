"""Set Cover: greedy H_n-approximation and exact OPT."""
from __future__ import annotations

from dataclasses import dataclass
import itertools
import random


@dataclass
class SetCoverInstance:
    universe: set[int]
    sets: list[set[int]]
    names: list[str] | None = None

    @property
    def m(self) -> int:
        return len(self.sets)


def greedy_set_cover(inst: SetCoverInstance) -> list[int]:
    """
    Greedy Set Cover: repeatedly pick the set covering the most uncovered elements.

    Guarantee: |approx| ≤ H_{|U|} · OPT  where H_n = 1 + 1/2 + ... + 1/n ≤ ln n + 1.
    Returns indices into inst.sets.
    """
    uncovered = set(inst.universe)
    chosen: list[int] = []
    remaining = set(range(inst.m))

    while uncovered:
        best_i = None
        best_gain = -1
        for i in remaining:
            gain = len(inst.sets[i] & uncovered)
            if gain > best_gain:
                best_gain = gain
                best_i = i
        if best_i is None or best_gain <= 0:
            raise ValueError("universe not fully coverable by given sets")
        chosen.append(best_i)
        uncovered -= inst.sets[best_i]
        remaining.remove(best_i)
    return chosen


def set_cover_exact(inst: SetCoverInstance) -> list[int]:
    """Brute-force minimum set cover (small m only)."""
    if inst.m > 20:
        raise ValueError("exact set cover only for m<=20")
    universe = inst.universe
    for k in range(0, inst.m + 1):
        for combo in itertools.combinations(range(inst.m), k):
            covered: set[int] = set()
            for i in combo:
                covered |= inst.sets[i]
            if universe <= covered:
                return list(combo)
    raise ValueError("no cover exists")


def is_set_cover(inst: SetCoverInstance, chosen: list[int]) -> bool:
    covered: set[int] = set()
    for i in chosen:
        covered |= inst.sets[i]
    return inst.universe <= covered


def harmonic(n: int) -> float:
    return sum(1.0 / i for i in range(1, n + 1)) if n > 0 else 0.0


def random_set_cover(
    n: int,
    m: int,
    *,
    set_size: int = 3,
    seed: int = 0,
) -> SetCoverInstance:
    """Random instance: universe {0..n-1}, m random subsets of given size."""
    rng = random.Random(seed)
    universe = set(range(n))
    sets: list[set[int]] = []
    for _ in range(m):
        k = min(set_size, n)
        sets.append(set(rng.sample(range(n), k)))
    covered: set[int] = set()
    for s in sets:
        covered |= s
    if covered != universe:
        sets.append(set(universe))
    return SetCoverInstance(universe=universe, sets=sets)


def adversarial_set_cover(*, seed: int = 0, tries: int = 400) -> SetCoverInstance:
    """
    Search for a small random instance where greedy is strictly worse than OPT.

    (Unweighted greedy is often near-optimal on tiny instances; we search until
    we find a gap, which is useful for demos.)
    """
    for t in range(tries):
        inst = random_set_cover(n=7, m=10, set_size=2, seed=seed + t)
        # drop the full-universe filler if present and still coverable without it
        if len(inst.sets[-1]) == len(inst.universe):
            trimmed = SetCoverInstance(universe=inst.universe, sets=inst.sets[:-1])
            covered: set[int] = set()
            for s in trimmed.sets:
                covered |= s
            if covered == trimmed.universe:
                inst = trimmed
        try:
            opt = set_cover_exact(inst)
            gre = greedy_set_cover(inst)
        except ValueError:
            continue
        if len(gre) > len(opt):
            return inst
    # Fallback hand instance (OPT=2, often greedy=2; still valid study case)
    return SetCoverInstance(
        universe={0, 1, 2, 3},
        sets=[{0, 1, 2}, {0, 1, 3}, {2, 3}, {0}, {1}, {2}, {3}],
    )
