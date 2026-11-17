"""Run approximation-vs-OPT studies on random and adversarial instances."""
from __future__ import annotations

from dataclasses import dataclass

from approx_algorithms.graph import (
    adversarial_vc_disjoint_edges,
    adversarial_vc_star,
    random_graph,
)
from approx_algorithms.set_cover import (
    adversarial_set_cover,
    greedy_set_cover,
    harmonic,
    random_set_cover,
    set_cover_exact,
)
from approx_algorithms.vertex_cover import vertex_cover_2approx, vertex_cover_exact


@dataclass
class TrialResult:
    name: str
    approx_size: int
    opt_size: int
    ratio: float
    guarantee: float


def _ratio(approx: int, opt: int) -> float:
    if opt <= 0:
        return 1.0 if approx == 0 else float("inf")
    return approx / opt


def study_vertex_cover(*, seed: int = 1) -> list[TrialResult]:
    results: list[TrialResult] = []

    # Random G(n,p)
    for i, (n, p) in enumerate([(10, 0.25), (12, 0.2), (14, 0.15)]):
        g = random_graph(n, p, seed=seed + i)
        if not g.edges:
            continue
        approx = vertex_cover_2approx(g, seed=seed + i)
        opt = vertex_cover_exact(g)
        results.append(
            TrialResult(
                name=f"random G({n},{p})",
                approx_size=len(approx),
                opt_size=len(opt),
                ratio=_ratio(len(approx), len(opt)),
                guarantee=2.0,
            )
        )

    # Adversarial: star (OPT=1, approx=2)
    star = adversarial_vc_star(9)
    a = vertex_cover_2approx(star, seed=0)
    o = vertex_cover_exact(star)
    results.append(
        TrialResult(
            name="adversarial star (n=9)",
            approx_size=len(a),
            opt_size=len(o),
            ratio=_ratio(len(a), len(o)),
            guarantee=2.0,
        )
    )

    # Adversarial: disjoint edges (ratio → 2)
    dis = adversarial_vc_disjoint_edges(5)
    a = vertex_cover_2approx(dis, seed=1)
    o = vertex_cover_exact(dis)
    results.append(
        TrialResult(
            name="adversarial 5 disjoint edges",
            approx_size=len(a),
            opt_size=len(o),
            ratio=_ratio(len(a), len(o)),
            guarantee=2.0,
        )
    )
    return results


def study_set_cover(*, seed: int = 1) -> list[TrialResult]:
    results: list[TrialResult] = []

    for i, (n, m, ss) in enumerate([(6, 8, 2), (7, 10, 3), (8, 12, 3)]):
        inst = random_set_cover(n, m, set_size=ss, seed=seed + i)
        gre = greedy_set_cover(inst)
        opt = set_cover_exact(inst)
        H = harmonic(len(inst.universe))
        results.append(
            TrialResult(
                name=f"random n={n} m={len(inst.sets)} s≈{ss}",
                approx_size=len(gre),
                opt_size=len(opt),
                ratio=_ratio(len(gre), len(opt)),
                guarantee=H,
            )
        )

    adv = adversarial_set_cover(seed=seed)
    gre = greedy_set_cover(adv)
    opt = set_cover_exact(adv)
    results.append(
        TrialResult(
            name=f"adversarial search (|U|={len(adv.universe)}, m={adv.m})",
            approx_size=len(gre),
            opt_size=len(opt),
            ratio=_ratio(len(gre), len(opt)),
            guarantee=harmonic(len(adv.universe)),
        )
    )
    return results


def format_study(title: str, rows: list[TrialResult]) -> str:
    lines = [
        title,
        f"{'instance':<40} {'approx':>7} {'OPT':>5} {'ratio':>7} {'bound':>7}",
        "-" * 70,
    ]
    for r in rows:
        lines.append(
            f"{r.name:<40} {r.approx_size:7d} {r.opt_size:5d} {r.ratio:7.3f} {r.guarantee:7.3f}"
        )
    return "\n".join(lines)
