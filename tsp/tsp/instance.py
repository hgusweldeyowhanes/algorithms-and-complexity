"""TSP instance: cities as 2D points, Euclidean distances."""
from __future__ import annotations

import math
from dataclasses import dataclass
from pathlib import Path


@dataclass(frozen=True)
class City:
    name: str
    x: float
    y: float


@dataclass
class Instance:
    cities: list[City]

    @property
    def n(self) -> int:
        return len(self.cities)

    def dist(self, i: int, j: int) -> float:
        a, b = self.cities[i], self.cities[j]
        return math.hypot(a.x - b.x, a.y - b.y)

    def tour_length(self, tour: list[int]) -> float:
        if len(tour) < 2:
            return 0.0
        total = 0.0
        for a, b in zip(tour, tour[1:]):
            total += self.dist(a, b)
        total += self.dist(tour[-1], tour[0])
        return total

    def distance_matrix(self) -> list[list[float]]:
        n = self.n
        m = [[0.0] * n for _ in range(n)]
        for i in range(n):
            for j in range(i + 1, n):
                d = self.dist(i, j)
                m[i][j] = m[j][i] = d
        return m


def load_instance(path: str | Path) -> Instance:
    """
    Load a simple text instance:

        name x y
        ...

    Lines starting with # are comments. Blank lines ignored.
    """
    p = Path(path)
    text = p.read_text(encoding="utf-8", errors="replace")
    cities: list[City] = []
    for lineno, raw in enumerate(text.splitlines(), start=1):
        line = raw.strip()
        if not line or line.startswith("#"):
            continue
        parts = line.split()
        if len(parts) < 3:
            raise ValueError(f"{p}:{lineno}: expected 'name x y'")
        name = parts[0]
        try:
            x = float(parts[1])
            y = float(parts[2])
        except ValueError as exc:
            raise ValueError(f"{p}:{lineno}: invalid coordinates") from exc
        cities.append(City(name=name, x=x, y=y))
    if len(cities) < 2:
        raise ValueError("need at least 2 cities")
    return Instance(cities=cities)
