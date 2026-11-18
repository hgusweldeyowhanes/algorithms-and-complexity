"""NP-completeness reductions with bidirectional solution verification."""

from np_reductions.sat import Formula3SAT, parse_formula
from np_reductions.clique import reduce_3sat_to_clique
from np_reductions.vertex_cover import reduce_3sat_to_vertex_cover

__all__ = [
    "Formula3SAT",
    "parse_formula",
    "reduce_3sat_to_clique",
    "reduce_3sat_to_vertex_cover",
]
__version__ = "0.1.0"
