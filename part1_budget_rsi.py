"""
PART 1 - Budget RSI (arXiv:2610.29121)            course labs 10.03, 10.04, 10.05

   10.03  Run the scorer   : fit a linear readout from 96 labelled patches, compare
                             random vs parent-rate-ablation vs readout, count the cost
   10.04  Analyse it       : learned coefficients, feature ablation, training-set size
   10.05  Extend           : score real held-out bugs; how often is the TRUE fix ranked first?

Run:  python part1_budget_rsi.py
"""
import random
import statistics as st

import numpy as np

from rsi_debug.agent import (FEATURE_NAMES, N_FEAT, Harness, Policy, featurize,
                             ochiai)
from rsi_debug.corpus import (apply_edit, candidates, pass_rate, run_tests)
from rsi_debug.evaluate import evaluate, load_bugs
from rsi_debug.scorer import collect_labeled_edits, fit_readout


def banner(t):
    print("\n" + "=" * 78 + f"\n{t}\n" + "=" * 78)


train, held = load_bugs()
SEEDS = range(10)

# ---------------------------------------------------------------------------
banner("10.03  Run the Budget RSI scorer")
print("Step 1. Collect 96 labelled patches on TRAINING bugs.")
print("        label = how much the patch raises the visible-CI pass rate.")
X, y, fit_cost = collect_labeled_edits(train, n=96, seed=0)
print(f"        96 patches  -> {int((y > 0).sum())} improved CI, {int((y < 0).sum())} made it worse, "
      f"{int((y == 0).sum())} changed nothing")
print(f"        one-time labelling cost: {fit_cost} test executions")

print("\nStep 2. Fit the linear readout   score = w . [parent_rate + 8 patch features]")
readout = fit_readout(X, y)
ablation = fit_readout(X, y, cols=[0])       # parent-rate only: cannot rank siblings
print("        done (ridge regression, pure NumPy)")

print("\nStep 3. Same agent loop, same budget (40 test runs/bug), only the SELECTOR differs.")
print("        All use: accept only strict CI improvement, never re-run a tried patch.\n")
rows = {}
for k in (1, 2, 3):
    hr = Harness(k=k, accept="gt", dedupe=True)
    hl = Harness(k=k, accept="gt", dedupe=True, use_loc=True)
    rows[("random", k)] = evaluate(f"random          k={k}", held, hr, Policy(), seeds=SEEDS)
    rows[("ablation", k)] = evaluate(f"parent-rate     k={k}", held, hl, ablation, seeds=SEEDS)
    rows[("readout", k)] = evaluate(f"BUDGET RSI      k={k}", held, hl, readout, seeds=SEEDS)
    for name in ("random", "ablation", "readout"):
        print(rows[(name, k)].row())
    print()

print("Cost view: full evaluation would run CI on EVERY candidate patch each round.")
n_c = st.mean(len(candidates(b.buggy_src)) for b in held)
print(f"   candidates per bug: {n_c:.1f}   vs  Budget RSI evaluates k=1..3 per round")
for k in (1, 2, 3):
    print(f"   k={k}: patches executed per round cut by {100 * (1 - k / n_c):.0f}%")
print(f"   but the scorer is not free: {fit_cost} executions to label, amortised over the "
      f"{len(held)} held-out bugs = {fit_cost / len(held):.0f} executions/bug")

# ---------------------------------------------------------------------------
banner("10.04  Analyse the scoring mechanism")
print("Learned coefficients (largest magnitude first):")
for n_, w in sorted(zip(FEATURE_NAMES, readout.w), key=lambda t: -abs(t[1])):
    print(f"   {n_:<14} {w:+.3f}")
print("\nReading: operator type carries the signal; 'ochiai' (fault localisation) is ~0.")
print("That is a finding, not a bug: with only 25 improving patches in 96, a LINEAR model")
print("on one-hot operators learns 'which patch KINDS tend to help', not 'where the fault is'.")

print("\nAblation A - drop features (held-out fix rate, k=1, 10 independent fit seeds):")
sets = {
    "all 9 features": list(range(N_FEAT)),
    "no ochiai/rel_line (7)": list(range(0, 7)),
    "ops only, no parent (6)": list(range(1, 7)),
    "parent_rate only (1)": [0],
}
for label, cols in sets.items():
    vals = []
    for fs in range(10):
        Xf, yf, _ = collect_labeled_edits(train, 96, seed=100 + fs)
        pol = fit_readout(Xf, yf, cols=cols)
        vals.append(evaluate("", held, Harness(k=1, accept="gt", dedupe=True, use_loc=True), pol,
                             seeds=range(3)).fix_rate)
    print(f"   {label:<26} fixed {100 * st.mean(vals):5.1f}%  (+/-{100 * st.pstdev(vals):.1f} across fits)")

print("\nAblation B - number of labelled patches (k=1, 10 fit seeds each):")
for n in (12, 24, 48, 96, 192):
    vals = []
    for fs in range(10):
        Xf, yf, _ = collect_labeled_edits(train, n, seed=200 + fs)
        pol = fit_readout(Xf, yf)
        vals.append(evaluate("", held, Harness(k=1, accept="gt", dedupe=True, use_loc=True), pol,
                             seeds=range(3)).fix_rate)
    print(f"   {n:>3} labelled patches   fixed {100 * st.mean(vals):5.1f}%  (+/-{100 * st.pstdev(vals):.1f})")

# ---------------------------------------------------------------------------
banner("10.05  Extend: does the scorer rank a VALID fix highly on unseen bugs?")
print("For every single-fault held-out bug: score all candidate patches (zero CI runs) and find")
print("where the FIRST patch that really fixes the bug (passes the hidden tests) lands.")
print("(Several patches can be valid fixes, so we do not require the exact injected inverse.)\n")
from rsi_debug.corpus import is_fixed
ranks, rand_exp, hit1, hit3, n_bug = [], [], 0, 0, 0
for b in held:
    if b.n_faults != 1:
        continue
    res = run_tests(b.buggy_src, b.visible)
    cands = candidates(b.buggy_src)
    Xb = featurize(cands, pass_rate(res), ochiai(res), b.buggy_src.count("\n") + 1)
    sc = Xb @ readout.w
    order = list(np.argsort(-sc, kind="stable"))
    valid = [i for i, e in enumerate(cands) if is_fixed(b, apply_edit(b.buggy_src, e))]
    r = min(order.index(i) for i in valid) + 1
    ranks.append(r)
    rand_exp.append((len(cands) + 1) / (len(valid) + 1))   # expected first-hit rank, random order
    hit1 += r == 1
    hit3 += r <= 3
    n_bug += 1
print(f"single-fault held-out bugs: {n_bug}")
print(f"valid fix ranked #1 : {hit1}/{n_bug} = {100*hit1/n_bug:.0f}%")
print(f"valid fix in top-3  : {hit3}/{n_bug} = {100*hit3/n_bug:.0f}%")
print(f"mean rank of first valid fix: {st.mean(ranks):.2f}   (random order expectation {st.mean(rand_exp):.2f})")
print("-> the one-shot ranking edge is small; the end-to-end gain comes from it compounding over rounds.")

print("\nWhere does the end-to-end gain come from? Fix rate by held-out function (k=1, 10 seeds):")
from collections import defaultdict
import random as _r
hl = Harness(k=1, accept="gt", dedupe=True, use_loc=True)
hr = Harness(k=1, accept="gt", dedupe=True)
from rsi_debug.agent import run_episode
by = {"readout": defaultdict(list), "random": defaultdict(list)}
for s in SEEDS:
    for name, h, pol in (("readout", hl, readout), ("random", hr, Policy())):
        rng = _r.Random(1000 + s)
        for b in held:
            by[name][b.func].append(run_episode(b, h, pol, rng).fixed)
print(f"   {'function':<14}{'random':>8}{'readout':>9}")
for f in sorted(by["random"]):
    print(f"   {f:<14}{100*st.mean(by['random'][f]):>7.0f}%{100*st.mean(by['readout'][f]):>8.0f}%")
print("   (the scorer helps on some functions and does nothing - or slightly hurts - on 'leap')")

print("\nWorked example (one held-out bug): predicted score vs actual CI gain")
b = next(b for b in held if b.n_faults == 1 and b.func == "inversions")
res = run_tests(b.buggy_src, b.visible)
cands = candidates(b.buggy_src)
Xb = featurize(cands, pass_rate(res), ochiai(res), b.buggy_src.count("\n") + 1)
sc = Xb @ readout.w
print(b.buggy_src)
print(f"   {'patch':<16}{'score':>8}{'actual CI gain':>16}   note")
for i in np.argsort(-sc):
    e = cands[i]
    new_src = apply_edit(b.buggy_src, e)
    gain = pass_rate(run_tests(new_src, b.visible)) - pass_rate(res)
    print(f"   {str(e):<16}{sc[i]:>8.3f}{gain:>+16.2f}   {'<- valid fix' if is_fixed(b, new_src) else ''}")

print("\nHONEST LIMITS: operator priors transfer across functions, localisation does not help")
print("in a linear readout, and the 96-patch fit is a real cost.  The gain over random is")
print("modest and uneven; it is not the paper's headline number (different task and scale).")
