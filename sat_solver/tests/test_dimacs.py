from __future__ import annotations

import pytest

from sat_solver.dimacs import DimacsError, parse_dimacs


def test_parse_basic():
    f = parse_dimacs(
        """
c comment
p cnf 2 2
1 -2 0
-1 2 0
"""
    )
    assert f.num_vars == 2
    assert len(f.clauses) == 2
    assert f.clauses[0].lits == [1, -2]


def test_parse_extra_spaces_and_multiline_clause():
    f = parse_dimacs(
        """
p cnf 3 1
1  2
  -3 0
"""
    )
    assert f.clauses[0].lits == [1, 2, -3]


def test_parse_rejects_bad_header():
    with pytest.raises(DimacsError):
        parse_dimacs("p sat 1 1\n1 0\n")


def test_parse_rejects_literal_out_of_range():
    with pytest.raises(DimacsError):
        parse_dimacs("p cnf 1 1\n2 0\n")


def test_parse_rejects_unterminated():
    with pytest.raises(DimacsError):
        parse_dimacs("p cnf 1 1\n1\n")
