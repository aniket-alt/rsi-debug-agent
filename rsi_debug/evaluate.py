"""evaluate.py - run an agent configuration over many bugs and seeds, report stats."""
from __future__ import annotations

import random
import statistics as st
from dataclasses import dataclass

from .agent import Harness, Memory, Policy, run_episode
from .corpus import HELDOUT_FUNCS, TRAIN_FUNCS, make_bugs

_CACHE = {}


def load_bugs():
    if "b" not in _CACHE:
        _CACHE["b"] = (make_bugs(TRAIN_FUNCS, seed=0), make_bugs(HELDOUT_FUNCS, seed=1))
    return _CACHE["b"]


@dataclass
class Summary:
    name: str
    fix_rate: float
    fix_sd: float          # sd across seeds
    cost_per_bug: float
    thrash_per_bug: float
    overfit_rate: float
    mismatches: float
    prevented: float
    n: int

    def row(self):
        return (f"{self.name:<34} fixed {self.fix_rate*100:5.1f}% (+/-{self.fix_sd*100:3.1f})  "
                f"overfit {self.overfit_rate*100:4.1f}%  cost/bug {self.cost_per_bug:5.1f}  "
                f"thrash/bug {self.thrash_per_bug:4.2f}  prevented/bug {self.prevented:4.2f}")


def per_bug(bugs, harness: Harness, policy: Policy, memory: Memory | None = None,
            seeds=range(10)):
    """Mean fix indicator per bug (averaged over seeds) - used for bootstrap CIs."""
    import numpy as np
    acc = np.zeros(len(bugs))
    for s in seeds:
        rng = random.Random(1000 + s)
        for i, b in enumerate(bugs):
            acc[i] += run_episode(b, harness, policy, rng,
                                  memory=memory.copy() if memory else None).fixed
    return acc / len(list(seeds))


def evaluate(name, bugs, harness: Harness, policy: Policy, memory: Memory | None = None,
             seeds=range(10), budget=None) -> Summary:
    kw = {} if budget is None else {"budget": budget}
    per_seed, eps_all = [], []
    for s in seeds:
        rng = random.Random(1000 + s)
        eps = [run_episode(b, harness, policy, rng, memory=memory.copy() if memory else None,
                           learn=False, **kw) for b in bugs]
        per_seed.append(sum(e.fixed for e in eps) / len(eps))
        eps_all += eps
    n = len(eps_all)
    return Summary(
        name, st.mean(per_seed), st.pstdev(per_seed),
        sum(e.cost for e in eps_all) / n,
        sum(e.thrash for e in eps_all) / n,
        sum(e.reason == "overfit" for e in eps_all) / n,
        sum(e.mismatches for e in eps_all) / n,
        sum(e.prevented for e in eps_all) / n, n)
