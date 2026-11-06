"""Smallest complete example: Re = 100 cavity on 64 x 64 cells, compared with Ghia et al."""
import os
import sys

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
from simplefvm import SimpleSolver, ghia  # noqa: E402

s = SimpleSolver(M=64, Re=100)              # square cavity, flat lid
s.run(tol=1e-6, verbose=200)
print(f"converged={s.converged} after {s.iterations} iterations in {s.wall_time:.1f} s")
print("deviation from Ghia et al. (1982):", ghia.compare(s, 100))

# A curved lid uses the same call; 'quadratic', 'linear' and 'crazy' are built in.
c = SimpleSolver(M=48, Re=100, H1=2.0, H2=1.2, shape="quadratic")
c.run(tol=1e-6, verbose=0)
print(f"curved lid: converged={c.converged}, max cell divergence {c.max_divergence():.1e}")

# Extra transported scalars (e.g. mixture fraction for a flamelet model) go in fields.vars:
s.fields.add_field("Z", 0.0)
