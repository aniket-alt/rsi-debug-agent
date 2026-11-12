"""
PART 4 - Fair comparison and an honest claim          (course themes 09.03-09.07, 11.02)

All arms: same 43 held-out bugs (5 functions never used for fitting/tuning), same budget
(40 test executions per bug), same evaluation seeds.  Differences are therefore due to the
method, not to extra compute at inference time.

We bootstrap over BUGS (not just seeds) because the bug set is the main source of variance.

Prereq: run part3_self_hpo.py first (it saves outputs/part3_results.json).
Run:    python part4_final_comparison.py
"""
import json
import os
import random

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np

from rsi_debug.agent import Harness, Memory, Policy, run_episode
from rsi_debug.evaluate import load_bugs, per_bug
from rsi_debug.scorer import collect_labeled_edits, fit_readout
from rsi_debug.self_hpo import h_step, p_step


def banner(t):
    print("\n" + "=" * 78 + f"\n{t}\n" + "=" * 78)


train, held = load_bugs()
SEEDS = range(10)
rng_boot = np.random.default_rng(0)

# ---------------------------------------------------------------- build every arm
Xf, yf, _ = collect_labeled_edits(train, 96, seed=0)
readout = fit_readout(Xf, yf)

memory = Memory()                                   # SO-RSI memory, warmed on TRAIN bugs only
for p in range(3):
    for b in train:
        run_episode(b, Harness(so_rsi=True), Policy(), random.Random(p), memory=memory, learn=True)

arms = {}
arms["Naive (first-order)"] = per_bug(held, Harness(), Policy(), None, SEEDS)
arms["Budget RSI"] = per_bug(held, Harness(k=1, accept="gt", dedupe=True, use_loc=True), readout, None, SEEDS)
arms["SO-RSI"] = per_bug(held, Harness(so_rsi=True), Policy(), memory, SEEDS)
arms["Budget RSI + SO-RSI"] = per_bug(
    held, Harness(k=1, accept="gt", use_loc=True, so_rsi=True), readout, memory, SEEDS)

# Self-HPO from part 3 (per-bug vectors, one per training seed)
if os.path.exists("outputs/part3_results.json"):
    r3 = json.load(open("outputs/part3_results.json"))
    arms["Self-HPO (from scratch)"] = np.mean(r3["JOINT"], axis=0)
    n3 = len(r3["JOINT"])
else:
    print("WARNING: outputs/part3_results.json missing - run part3_self_hpo.py first")
    n3 = 0

# Self-HPO whose policy STARTS from the Budget RSI readout (all three methods chained)
print("training Self-HPO with a Budget-RSI-initialised policy (3 seeds x 4 cycles) ...")
vecs = []
for seed in (0, 1, 2):
    rng = random.Random(seed)
    mem, h, p = Memory(), Harness(), readout.copy()
    for c in range(4):
        h, *_ = h_step(p, h, train, mem, rng)
        p, _ = p_step(p, h, train, mem, rng)
    d = p.copy()
    d.mode = "sample"
    vecs.append(per_bug(held, h, d, mem, SEEDS))
    print(f"   seed {seed}: final harness [{h}]")
arms["Self-HPO + Budget RSI init"] = np.mean(vecs, axis=0)

# ---------------------------------------------------------------- statistics
base = arms["Naive (first-order)"]


def boot(v, base=None, n=4000):
    idx = rng_boot.integers(0, len(v), (n, len(v)))
    stat = (v[idx] - base[idx]).mean(1) if base is not None else v[idx].mean(1)
    return np.percentile(stat, [2.5, 97.5])


banner("RESULT - held-out bugs fixed (hidden tests pass), 95% bootstrap CI over bugs")
print(f"{'method':<30}{'fixed':>8}   {'95% CI':<16}{'gain vs naive':>14}   {'95% CI of gain':<18}")
for name, v in arms.items():
    lo, hi = boot(v)
    if name.startswith("Naive"):
        print(f"{name:<30}{100*v.mean():>7.1f}%   [{100*lo:4.1f}, {100*hi:4.1f}]")
    else:
        glo, ghi = boot(v, base)
        mark = "  *" if glo > 0 else ""
        print(f"{name:<30}{100*v.mean():>7.1f}%   [{100*lo:4.1f}, {100*hi:4.1f}]  "
              f"{100*(v.mean()-base.mean()):>+9.1f} pts   [{100*glo:+5.1f}, {100*ghi:+5.1f}]{mark}")
print("\n * = the 95% CI of the paired gain over naive excludes zero.")
print(f"(Self-HPO rows average {n3 if n3 else 3} independent training seeds; the others are deterministic given the seeds.)")

# ---------------------------------------------------------------- figure
fig, ax = plt.subplots(figsize=(7, 3.8))
names = list(arms)
means = [100 * arms[n].mean() for n in names]
err = np.array([[100 * arms[n].mean() - 100 * boot(arms[n])[0], 100 * boot(arms[n])[1] - 100 * arms[n].mean()]
                for n in names]).T
cols = ["#8e9aaf"] + ["#2a9d8f"] * (len(names) - 1)
ax.barh(names[::-1], means[::-1], xerr=err[:, ::-1], color=cols[::-1], capsize=3)
for i, m in enumerate(means[::-1]):
    ax.text(m + 1.2, i, f"{m:.0f}%", va="center", fontsize=9)
ax.set_xlabel("held-out bugs fixed (%)  - 40 test runs per bug, 43 bugs from 5 unseen functions")
ax.set_xlim(0, 100)
ax.set_title("Debugging-agent RSI methods, matched budget")
plt.tight_layout()
plt.savefig("outputs/part4_comparison.png", dpi=130)
print("saved outputs/part4_comparison.png")

# ---------------------------------------------------------------- claim
banner("THE CLAIM, WITH ITS LIMITS  (course lab 09.07)")
print("""THREE OBJECTS (lab 09.01)
  task model       : the debugging agent (CI -> patch -> CI loop, patch scorer, budget 40 runs)
  research skill   : the harness (k patches/round, extra tests, localisation, SO-RSI on, accept rule)
  improver         : Self-HPO (H-step rewrites the harness, P-step rewrites the patch policy)
  What recursed    : in Self-HPO the SAME policy that debugs also analyses its failures to
                     propose the next harness; each cycle inherits the previous harness and policy.

WHAT WAS TESTED
  Do Budget RSI, SO-RSI and Self-HPO each help a debugging agent fix held-out bugs under an
  identical test-execution budget?  Evidence: the table above (paired, bootstrap over bugs).

WHAT THE EVIDENCE SUPPORTS (read the CIs, not just the means)
  - Methods marked * beat the naive agent on these held-out bugs at the same budget.
  - Self-HPO's gain is mostly the H-step discovering that SO-RSI diagnostics + strict
    acceptance should be switched on; the policy-gradient step adds little on top, and the
    paper's 'joint > sum of parts' synergy was NOT reproduced (see part3_output.txt).
  - The confidence intervals of the starred methods overlap heavily.  We can say each beats
    naive; we can NOT rank Budget RSI, Budget RSI + SO-RSI and Self-HPO against each other.
    The simplest method (Budget RSI alone) scores at least as high as the more elaborate ones.

WHAT WAS NOT TESTED / DOES NOT FOLLOW
  - Bugs are synthetic single/double-operator mutations of 14 tiny functions, not real defects.
  - The agent is a search over a fixed patch vocabulary, not an LLM; no natural-language reasoning.
  - Only 5 held-out functions; one ('leap') is rarely solved by any method (<=7% in Part 1).
  - The H-step 'analysis' is rule-based, not an LLM proposer; the harness space is 5 knobs.
  - Setup costs are not in the per-bug budget: 960 runs (Budget RSI fit), memory warm-up,
    and Self-HPO training on 79 bugs.
  - Budget is fixed at 40 runs; conclusions may change at other budgets.
  - Laptop adaptations preserve each paper's IDEA, not its scale or headline numbers.
""")
