# NP-Completeness Reduction Visualizer

Implement classic **polynomial-time reductions** and **verify solutions in both directions**:

| Reduction | Target question |
|-----------|-----------------|
| **3-SAT → Clique** | Does \(G\) have a clique of size \(k=m\)? |
| **3-SAT → Vertex Cover** | Does \(G\) have a vertex cover of size \(k=n+2m\)? |

For a concrete formula \(\varphi\):

1. Build the gadget graph \(G\)
2. If \(\varphi\) is SAT, map a witness assignment → clique / cover
3. Map that combinatorial solution back → satisfying assignment
4. Also search the graph directly and check the reverse map

This shows the *theory* content of NP-completeness: yes-instances map to yes-instances, and certificates transform efficiently.

## Usage

```bash
cd np_reductions

python -m np_reductions
python -m np_reductions clique --graph
python -m np_reductions vertex-cover --file examples/sat_demo.txt
python -m np_reductions both --unsat-demo
python -m np_reductions --formula "(x ∨ ~y ∨ z) ∧ (~x ∨ y ∨ ~z)"
```

## Constructions (sketch)

**Clique:** one vertex per literal occurrence; edges between *consistent* literals from *different* clauses. Clique of size \(m\) picks one literal per clause → assignment.

**Vertex Cover:** edge \(\{x,\neg x\}\) per variable; a triangle per clause linked to the matching literal vertices. Cover size \(k=n+2m\).

## Library API

```python
from np_reductions import parse_formula, reduce_3sat_to_clique
from np_reductions.verify import verify_clique_reduction

phi = parse_formula("(x ∨ ~y ∨ z) ∧ (~x ∨ y ∨ z)")
red, check = verify_clique_reduction(phi)
print(check.ok, check.messages)
```

## Layout

```
np_reductions/   # package
examples/
tests/
```

## Note

Graph search is brute-force (teaching-size formulas only). The reductions themselves are the polynomial constructions you would describe on an exam or in a paper.
