"""CDCL SAT solver: 2-watched literals, 1-UIP learning, VSIDS, Luby restarts, phase saving."""
from __future__ import annotations

from dataclasses import dataclass
from typing import Iterable

from sat_solver.types import Clause, Formula, neg, var_of


@dataclass
class SolveResult:
    sat: bool
    model: dict[int, bool] | None = None  # var -> value

    def assignment_lits(self, num_vars: int) -> list[int]:
        if not self.sat or self.model is None:
            return []
        out: list[int] = []
        for v in range(1, num_vars + 1):
            val = self.model.get(v, False)
            out.append(v if val else -v)
        return out


def luby(i: int) -> int:
    """Luby sequence L_i for i >= 1: 1,1,2,1,1,2,4, ..."""
    k = 0
    while True:
        if i == (1 << (k + 1)) - 1:
            return 1 << k
        if (1 << k) <= i < (1 << (k + 1)) - 1:
            return luby(i - (1 << k) + 1)
        k += 1
        if k > 60:
            return 1


class Solver:
    def __init__(self, formula: Formula, *, seed: int | None = None) -> None:
        self.num_vars = formula.num_vars
        self.clauses: list[Clause] = []
        self.watches: dict[int, list[int]] = {}  # lit -> clause indices
        self.value: list[int] = [0] * (self.num_vars + 1)  # 0 unassigned, 1 True, -1 False
        self.level: list[int] = [0] * (self.num_vars + 1)
        self.reason: list[int | None] = [None] * (self.num_vars + 1)  # clause index or None
        self.trail: list[int] = []  # assigned literals
        self.trail_lim: list[int] = []  # trail index where each decision level starts
        self.qhead = 0
        self.decision_level = 0

        self.activity = [0.0] * (self.num_vars + 1)
        self.var_inc = 1.0
        self.var_decay = 0.95
        self.clause_inc = 1.0
        self.clause_decay = 0.999

        self.phase: list[int] = [1] * (self.num_vars + 1)  # preferred polarity as lit sign (+1 / -1)
        self.rng = _Rng(seed if seed is not None else 1)

        self.conflicts = 0
        self.restarts = 0
        self.restart_count = 1  # Luby index
        self.conflict_limit = 100 * luby(1)
        self.learnt_limit = max(100, len(formula.clauses) // 3)

        for cl in formula.clauses:
            if cl.is_tautology:
                continue
            self._add_clause(cl.lits, learnt=False)

    # ----- public API -----

    def solve(self) -> SolveResult:
        if any(len(c.lits) == 0 for c in self.clauses if not c.is_tautology):
            return SolveResult(False, None)

        # Attach units / empty already handled
        while True:
            conflict = self._propagate()
            if conflict is not None:
                self.conflicts += 1
                if self.decision_level == 0:
                    return SolveResult(False, None)
                learnt, back_level, asserting = self._analyze(conflict)
                self._cancel_until(back_level)
                if not learnt:
                    # should not happen at dl>0
                    return SolveResult(False, None)
                ci = self._add_clause(learnt, learnt=True)
                self._enqueue(asserting, ci)
                self._var_decay_activity()
                self._clause_decay_activity()

                if self.conflicts >= self.conflict_limit:
                    self._restart()
                    self.restart_count += 1
                    self.conflict_limit = self.conflicts + 100 * luby(self.restart_count)
                    self._reduce_db()
            else:
                if self._num_assigned() >= self.num_vars:
                    return SolveResult(True, self._build_model())
                if not self._decide():
                    return SolveResult(True, self._build_model())

    # ----- clause attach -----

    def _add_clause(self, lits: Iterable[int], *, learnt: bool) -> int | None:
        lits = list(lits)
        # filter falsified / satisfied at dl0 only carefully — keep raw for CDCL
        seen: set[int] = set()
        cleaned: list[int] = []
        for lit in lits:
            if lit in seen:
                continue
            if neg(lit) in seen:
                return None  # tautology
            seen.add(lit)
            cleaned.append(lit)

        if not cleaned:
            # empty clause
            ci = len(self.clauses)
            self.clauses.append(Clause([], learnt=learnt))
            return ci

        if len(cleaned) == 1:
            ci = len(self.clauses)
            cl = Clause(cleaned, learnt=learnt)
            self.clauses.append(cl)
            lit = cleaned[0]
            v = var_of(lit)
            if self.value[v] == 0:
                self._enqueue(lit, ci)
            elif self._lit_false(lit):
                # Conflicting unit at current level → empty clause conflict
                empty_i = len(self.clauses)
                self.clauses.append(Clause([], learnt=False))
                return empty_i
            return ci

        ci = len(self.clauses)
        cl = Clause(cleaned, learnt=learnt)
        if learnt:
            cl.activity = self.clause_inc
        self.clauses.append(cl)
        self._attach(ci)
        if learnt:
            self._bump_clause(ci)
        return ci

    def _attach(self, ci: int) -> None:
        cl = self.clauses[ci]
        if len(cl.lits) < 2:
            return
        for lit in (cl.lits[0], cl.lits[1]):
            self.watches.setdefault(lit, []).append(ci)

    def _detach(self, ci: int) -> None:
        cl = self.clauses[ci]
        if len(cl.lits) < 2:
            return
        for lit in (cl.lits[0], cl.lits[1]):
            ws = self.watches.get(lit)
            if not ws:
                continue
            try:
                ws.remove(ci)
            except ValueError:
                pass

    # ----- assignment -----

    def _lit_true(self, lit: int) -> bool:
        v = var_of(lit)
        val = self.value[v]
        if val == 0:
            return False
        return (val > 0) == (lit > 0)

    def _lit_false(self, lit: int) -> bool:
        v = var_of(lit)
        val = self.value[v]
        if val == 0:
            return False
        return (val > 0) != (lit > 0)

    def _enqueue(self, lit: int, reason: int | None) -> bool:
        v = var_of(lit)
        if self.value[v] != 0:
            return not self._lit_false(lit)
        self.value[v] = 1 if lit > 0 else -1
        self.level[v] = self.decision_level
        self.reason[v] = reason
        self.phase[v] = 1 if lit > 0 else -1
        self.trail.append(lit)
        return True

    def _propagate(self) -> int | None:
        """Return conflicting clause index, or None."""
        while self.qhead < len(self.trail):
            p = self.trail[self.qhead]
            self.qhead += 1
            false_lit = neg(p)
            ws = self.watches.get(false_lit)
            if not ws:
                continue
            i = 0
            while i < len(ws):
                ci = ws[i]
                cl = self.clauses[ci]
                # Ensure false_lit is at index 1 (or 0)
                if cl.lits[0] == false_lit:
                    cl.lits[0], cl.lits[1] = cl.lits[1], cl.lits[0]
                # Now lits[1] == false_lit
                first = cl.lits[0]
                if self._lit_true(first):
                    i += 1
                    continue
                # Look for another non-false literal to watch
                found = False
                for k in range(2, len(cl.lits)):
                    if not self._lit_false(cl.lits[k]):
                        cl.lits[1], cl.lits[k] = cl.lits[k], cl.lits[1]
                        ws[i] = ws[-1]
                        ws.pop()
                        self.watches.setdefault(cl.lits[1], []).append(ci)
                        found = True
                        break
                if found:
                    continue
                # No replacement — clause is unit or conflict under first
                if self._lit_false(first):
                    # conflict — keep watch
                    return ci
                # unit propagate first
                if not self._enqueue(first, ci):
                    return ci
                i += 1
        return None

    # ----- conflict analysis (1-UIP) -----

    def _analyze(self, conflict_ci: int) -> tuple[list[int], int, int]:
        """Return (learnt_clause, backjump_level, asserting_literal)."""
        seen: set[int] = set()
        learnt: list[int] = []
        counter = 0
        p: int | None = None
        trail_index = len(self.trail) - 1
        ci: int | None = conflict_ci

        while True:
            assert ci is not None
            cl = self.clauses[ci]
            if cl.learnt:
                self._bump_clause(ci)
            for lit in cl.lits:
                if p is not None and lit == neg(p):
                    continue
                v = var_of(lit)
                if v in seen or self.level[v] == 0:
                    continue
                seen.add(v)
                self._bump_var(v)
                if self.level[v] >= self.decision_level:
                    counter += 1
                else:
                    learnt.append(lit)

            # next reason on trail
            while True:
                lit = self.trail[trail_index]
                trail_index -= 1
                v = var_of(lit)
                if v in seen:
                    break
            p = lit
            ci = self.reason[var_of(p)]
            counter -= 1
            if counter <= 0:
                break

        asserting = neg(p)
        learnt.insert(0, asserting)

        # backjump level: max level among learnt[1:], or 0
        back = 0
        if len(learnt) > 1:
            # put second-highest level lit at position 1 for watching
            max_i = 1
            for i in range(2, len(learnt)):
                if self.level[var_of(learnt[i])] > self.level[var_of(learnt[max_i])]:
                    max_i = i
            learnt[1], learnt[max_i] = learnt[max_i], learnt[1]
            back = self.level[var_of(learnt[1])]

        return learnt, back, asserting

    def _cancel_until(self, level: int) -> None:
        while self.decision_level > level:
            # undo trail from last decision
            start = self.trail_lim.pop() if self.trail_lim else 0
            while len(self.trail) > start:
                lit = self.trail.pop()
                v = var_of(lit)
                self.value[v] = 0
                self.reason[v] = None
                self.level[v] = 0
            self.decision_level -= 1
        self.qhead = len(self.trail)

    def _restart(self) -> None:
        self.restarts += 1
        self._cancel_until(0)
        self.qhead = 0

    def _decide(self) -> bool:
        # pick unassigned var with highest activity
        best_v = 0
        best_act = -1.0
        for v in range(1, self.num_vars + 1):
            if self.value[v] != 0:
                continue
            act = self.activity[v]
            # slight noise for ties
            if act > best_act or (act == best_act and self.rng.random() < 0.5):
                best_act = act
                best_v = v
        if best_v == 0:
            return False
        lit = best_v if self.phase[best_v] > 0 else -best_v
        self.decision_level += 1
        self.trail_lim.append(len(self.trail))
        self._enqueue(lit, None)
        return True

    # ----- VSIDS -----

    def _bump_var(self, v: int) -> None:
        self.activity[v] += self.var_inc
        if self.activity[v] > 1e100:
            for i in range(1, self.num_vars + 1):
                self.activity[i] *= 1e-100
            self.var_inc *= 1e-100

    def _var_decay_activity(self) -> None:
        self.var_inc /= self.var_decay

    def _bump_clause(self, ci: int) -> None:
        cl = self.clauses[ci]
        cl.activity += self.clause_inc
        if cl.activity > 1e20:
            for c in self.clauses:
                if c.learnt:
                    c.activity *= 1e-20
            self.clause_inc *= 1e-20

    def _clause_decay_activity(self) -> None:
        self.clause_inc /= self.clause_decay

    def _reduce_db(self) -> None:
        learnt = [(i, c) for i, c in enumerate(self.clauses) if c.learnt and len(c.lits) > 2]
        if len(learnt) <= self.learnt_limit:
            return
        learnt.sort(key=lambda t: t[1].activity)
        # delete half of low-activity clauses that are not reasons
        reasons = {r for r in self.reason if r is not None}
        remove_n = len(learnt) // 2
        for i, _ in learnt[:remove_n]:
            if i in reasons:
                continue
            self._detach(i)
            self.clauses[i] = Clause([1, -1], learnt=True)  # tautology placeholder
            self.clauses[i]._tautology = True  # type: ignore[attr-defined]

    # ----- helpers -----

    def _num_assigned(self) -> int:
        return sum(1 for v in range(1, self.num_vars + 1) if self.value[v] != 0)

    def _build_model(self) -> dict[int, bool]:
        return {v: self.value[v] > 0 for v in range(1, self.num_vars + 1) if self.value[v] != 0}


class _Rng:
    """Tiny LCG for tie-breaking (no numpy dependency)."""

    def __init__(self, seed: int) -> None:
        self.state = seed & 0xFFFFFFFF

    def random(self) -> float:
        self.state = (1103515245 * self.state + 12345) & 0x7FFFFFFF
        return self.state / 0x7FFFFFFF
