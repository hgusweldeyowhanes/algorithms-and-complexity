"""
3-SAT ≤p Vertex Cover

Construction (Garey–Johnson style):
  • For each variable x: edge {x, ~x}  (variable gadget).
  • For each clause C_i = (ℓ1 ∨ ℓ2 ∨ ℓ3): a triangle {c_i1, c_i2, c_i3}
    and edges connecting c_{ij} to the vertex for literal ℓj.

Target cover size: k = n + 2m
  (1 vertex from each variable edge + 2 from each clause triangle).

Claim: φ is satisfiable  ⟺  G has a vertex cover of size k.
"""
from __future__ import annotations

from dataclasses import dataclass

from np_reductions.graph import Graph
from np_reductions.sat import Formula3SAT, Literal


@dataclass
class VertexCoverReduction:
    formula: Formula3SAT
    graph: Graph
    k: int
    var_pos: dict[str, str]  # var -> vertex id for x
    var_neg: dict[str, str]  # var -> vertex id for ~x
    # clause triangle vertex -> (clause_index, literal)
    clause_vertex_lit: dict[str, tuple[int, Literal]]

    def assignment_to_cover(self, assignment: dict[str, bool]) -> set[str]:
        """Forward: satisfying assignment → vertex cover of size k."""
        if not self.formula.is_satisfied(assignment):
            raise ValueError("assignment does not satisfy the formula")
        cover: set[str] = set()
        # variable gadgets: pick the true literal's vertex
        for var, val in assignment.items():
            cover.add(self.var_pos[var] if val else self.var_neg[var])

        # clause triangles: leave out one true literal vertex; cover the other two
        for i, clause in enumerate(self.formula.clauses):
            true_js = [j for j, lit in enumerate(clause) if lit.eval(assignment)]
            if not true_js:
                raise ValueError(f"clause {i} unsatisfied")
            leave = true_js[0]
            for j in range(3):
                if j != leave:
                    cover.add(f"C{i}_{j}")
        if len(cover) != self.k:
            # if assignment leaves free vars unset, fill
            for var in self.formula.variables:
                if self.var_pos[var] not in cover and self.var_neg[var] not in cover:
                    cover.add(self.var_neg[var])
            # recompute clause part already done
        if len(cover) != self.k:
            raise ValueError(f"constructed cover has size {len(cover)}, expected k={self.k}")
        if not self.graph.is_vertex_cover(cover):
            raise ValueError("constructed set is not a vertex cover")
        return cover

    def cover_to_assignment(self, cover: set[str]) -> dict[str, bool]:
        """Backward: vertex cover of size ≤ k → satisfying assignment."""
        if len(cover) > self.k:
            raise ValueError(f"cover size {len(cover)} > k={self.k}")
        if not self.graph.is_vertex_cover(cover):
            raise ValueError("set is not a vertex cover")

        assignment: dict[str, bool] = {}
        for var in self.formula.variables:
            pos, neg = self.var_pos[var], self.var_neg[var]
            in_pos, in_neg = pos in cover, neg in cover
            # a min cover takes exactly one of {x,~x}; if both, prefer True
            if in_pos and not in_neg:
                assignment[var] = True
            elif in_neg and not in_pos:
                assignment[var] = False
            elif in_pos and in_neg:
                assignment[var] = True
            else:
                # neither — invalid for this construction's min covers, default False
                assignment[var] = False
        return assignment


def reduce_3sat_to_vertex_cover(formula: Formula3SAT) -> VertexCoverReduction:
    n = len(formula.variables)
    m = len(formula.clauses)
    k = n + 2 * m

    vertices: list[str] = []
    var_pos: dict[str, str] = {}
    var_neg: dict[str, str] = {}
    for var in formula.variables:
        vp, vn = f"x_{var}", f"nx_{var}"
        var_pos[var] = vp
        var_neg[var] = vn
        vertices.extend([vp, vn])

    clause_vertex_lit: dict[str, tuple[int, Literal]] = {}
    for i, clause in enumerate(formula.clauses):
        for j, lit in enumerate(clause):
            vid = f"C{i}_{j}"
            vertices.append(vid)
            clause_vertex_lit[vid] = (i, lit)

    g = Graph(vertices=vertices)
    for var in formula.variables:
        g.add_edge(var_pos[var], var_neg[var])
        g.meta[var_pos[var]] = var
        g.meta[var_neg[var]] = f"~{var}"

    for i, clause in enumerate(formula.clauses):
        a, b, c = f"C{i}_0", f"C{i}_1", f"C{i}_2"
        g.add_edge(a, b)
        g.add_edge(b, c)
        g.add_edge(a, c)
        for j, lit in enumerate(clause):
            cj = f"C{i}_{j}"
            g.meta[cj] = f"C{i}:{lit}"
            lit_v = var_pos[lit.var] if lit.positive else var_neg[lit.var]
            g.add_edge(cj, lit_v)

    return VertexCoverReduction(
        formula=formula,
        graph=g,
        k=k,
        var_pos=var_pos,
        var_neg=var_neg,
        clause_vertex_lit=clause_vertex_lit,
    )
