from __future__ import annotations

from pathlib import Path

from automata.loader import load_dfa, load_nfa, load_tm

ROOT = Path(__file__).resolve().parents[1]


def test_dfa_even_zeros():
    dfa = load_dfa(ROOT / "examples" / "dfa_even_zeros.json")
    assert dfa.accepts("") is True
    assert dfa.accepts("1") is True
    assert dfa.accepts("0") is False
    assert dfa.accepts("00") is True
    assert dfa.accepts("01") is False  # one zero
    assert dfa.accepts("010") is True  # two zeros
    ok, trace = dfa.run("01")
    assert ok is False
    assert trace[0] == "E"


def test_nfa_ends_with_ab():
    nfa = load_nfa(ROOT / "examples" / "nfa_ends_with_ab.json")
    assert nfa.accepts("ab") is True
    assert nfa.accepts("aab") is True
    assert nfa.accepts("babab") is True
    assert nfa.accepts("a") is False
    assert nfa.accepts("ba") is False
    assert nfa.accepts("") is False


def test_tm_anbn():
    tm = load_tm(ROOT / "examples" / "tm_anbn.json")
    assert tm.accepts("") is True
    assert tm.accepts("ab") is True
    assert tm.accepts("aabb") is True
    assert tm.accepts("aaabbb") is True
    assert tm.accepts("a") is False
    assert tm.accepts("ba") is False
    assert tm.accepts("abb") is False
    r = tm.run("aabb", keep_trace=True)
    assert r.accepted and r.halted
    assert r.steps >= 1
