# SAT Phase Transition

Experiment: **random k-SAT** undergoes a sharp **satisfiability phase transition** as the
clause-to-variable ratio \(\alpha = m/n\) grows.

For large random **3-SAT**, the threshold is near:

\[
\alpha_c \approx 4.26
\]

- Below \(\alpha_c\): instances are almost always **SAT**
- Above \(\alpha_c\): instances are almost always **UNSAT**
- Near \(\alpha_c\): hardest region for solvers

This project generates random formulas, solves them (CDCL from [`../sat_solver`](../sat_solver) when available), and reports \(P(\mathrm{SAT})\) vs \(\alpha\).

## Requirements

- Python 3.10+
- Sibling [`sat_solver`](../sat_solver) recommended (fallback DPLL is included)

## Usage

```bash
cd sat_phase_transition

# default: n=20, alpha 3.0..5.5, 11 steps, 20 trials each
python -m sat_phase_transition

# faster demo
python -m sat_phase_transition --n 15 --trials 10 --steps 9

# write CSV for plotting
python -m sat_phase_transition --n 18 --trials 15 --csv results.csv

# sample DIMACS formula near the threshold
python -m sat_phase_transition --demo-formula --n 20
```

### Typical output

```text
   alpha     m   P(SAT)   sat/trials     mean_s
------------------------------------------------
   3.000    60    1.000    20/20         0.0012
   ...
   4.250    85    0.450     9/20         0.0041
   ...
   5.500   110    0.000     0/20         0.0008
```

## Library API

```python
from sat_phase_transition import random_ksat, run_sweep

f = random_ksat(n=20, m=85, k=3, seed=1)
points = run_sweep(n=15, trials=10, steps=7)
for p in points:
    print(p.alpha, p.sat_probability)
```

## Layout

```
sat_phase_transition/   # package
tests/
```

## Notes

Finite-\(n\) curves are smooth; the jump sharpens as \(n\) grows. Use more `--trials` for less noisy estimates (slower).
