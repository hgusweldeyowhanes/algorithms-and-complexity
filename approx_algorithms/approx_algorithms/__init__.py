"""Approximation algorithms: Vertex Cover 2-approx and greedy Set Cover."""

from approx_algorithms.vertex_cover import (
    vertex_cover_2approx,
    vertex_cover_exact,
)
from approx_algorithms.set_cover import greedy_set_cover, set_cover_exact

__all__ = [
    "vertex_cover_2approx",
    "vertex_cover_exact",
    "greedy_set_cover",
    "set_cover_exact",
]
__version__ = "0.1.0"
