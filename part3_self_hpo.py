"""
PART 3 - Self-HPO: alternating harness and policy optimisation (arXiv:2610.22791)
                                                        course labs 10.12 and 10.14

   10.12  Compare workflow adaptation (RSIFlow vs Self-HPO) and justify the choice
   10.14  Implement the H-step / P-step alternation on the debugging agent and
          test the paper's claim: "evolving only one of the two is sub-optimal"

   H-step : the agent reads its OWN failure statistics and proposes harness edits
            (more tests? more patches per round? diagnostics? dedupe?).  The best
            harness on the HARD-bug set is kept.
   P-step : REINFORCE policy-gradient update of the patch-scoring policy under the
            chosen harness.

Arms (same number of cycles; the joint arm does both kinds of step each cycle):
   start   : default harness, untrained policy
   P-only  : only P-steps, harness frozen at default
   H-only  : only H-steps, policy frozen at untrained
   JOINT   : H-step then P-step each cycle  (Self-HPO)

Run:  python part3_self_hpo.py            (full: 3 training seeds x 4 cycles, ~several min)
      python part3_self_hpo.py --quick    (1 seed x 3 cycles)
"""
import random
import statistics as st
import sys

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np

from rsi_debug.agent import FEATURE_NAMES, Harness, Memory, Policy
import json

from rsi_debug.evaluate import evaluate, load_bugs, per_bug
from rsi_debug.self_hpo import h_step, p_step

QUICK = "--quick" in sys.argv
CYCLES = 3 if QUICK else 4
TRAIN_SEEDS = [0] if QUICK else [0, 1, 2]
EVAL_SEEDS = range(4) if QUICK else range(6)


def banner(t):
    print("\n" + "=" * 78 + f"\n{t}\n" + "=" * 78)


train, held = load_bugs()


def deploy(pol):
    d = pol.copy()
    d.mode = "sample"          # evaluate in the same stochastic mode the policy was trained in
    return d


def heldout(h, pol, mem):
    return evaluate("", held, h, deploy(pol), memory=mem, seeds=EVAL_SEEDS)


# ---------------------------------------------------------------------------
banner("10.12  RSIFlow vs Self-HPO: which adaptation fits a debugging agent?")
print("""                RSIFlow (2610.21791)              Self-HPO (2610.22791)
 what adapts    workflow orchestration            harness AND policy, alternately
 who proposes   an evolving meta-agent            the policy itself (H-step)
 trigger        the current performance bottleneck failures of the current policy
 fits us when   many pipeline stages to route     one agent loop + tunable test/CI harness
""")
print("Our agent is ONE loop (CI -> patch -> CI) with tunable harness knobs (how many tests,")
print("how many patches per round, diagnostics on/off).  That is Self-HPO's shape, so we use it.")
print("(RSIFlow's multi-stage router would need a multi-stage pipeline; lab 10.13 is skipped.)")

# ---------------------------------------------------------------------------
banner("10.14  Self-HPO on the debugging agent")
h0, p0 = Harness(), Policy()
base = heldout(h0, p0, None)
print(f"start (default harness, untrained policy): held-out fixed {100*base.fix_rate:.1f}%\n")

vec = {"P-only": [], "H-only": [], "JOINT": []}      # per-bug fix rates, saved for Part 4
curves = {"P-only": [], "H-only": [], "JOINT": []}
finals = {"P-only": [], "H-only": [], "JOINT": []}
details = {}

for seed in TRAIN_SEEDS:
    print(f"----- training seed {seed} -----")
    # ---------------- JOINT: H-step then P-step
    rng = random.Random(seed)
    mem, h, p = Memory(), h0, p0
    curve = []
    for c in range(1, CYCLES + 1):
        h, d, n_hard, tried = h_step(p, h, train, mem, rng)
        p, tr_fix = p_step(p, h, train, mem, rng)
        r = heldout(h, p, mem)
        curve.append(r.fix_rate)
        if seed == TRAIN_SEEDS[0]:
            print(f"[JOINT cycle {c}] failure diagnosis on train: "
                  f"exhausted {100*d['exhausted']:.0f}%  overfit {100*d['overfit']:.0f}%  "
                  f"thrash/bug {d['thrash']:.2f}   hard bugs: {n_hard}/{len(train)}")
            for cand, sc in tried:
                print(f"      proposed harness [{cand}]  hard-set fix rate {100*sc:.0f}%")
            print(f"      -> chosen harness [{h}]")
            print(f"      -> P-step done (train fix rate while learning {100*tr_fix:.0f}%) "
                  f"=> HELD-OUT fixed {100*r.fix_rate:.1f}%")
    curves["JOINT"].append(curve)
    finals["JOINT"].append(heldout(h, p, mem))
    details[("JOINT", seed)] = (h, p)
    vec["JOINT"].append(per_bug(held, h, deploy(p), mem, EVAL_SEEDS).tolist())

    # ---------------- P-only
    rng = random.Random(seed)
    mem, p = Memory(), p0
    curve = []
    for c in range(CYCLES):
        p, _ = p_step(p, h0, train, mem, rng)
        curve.append(heldout(h0, p, mem).fix_rate)
    curves["P-only"].append(curve)
    finals["P-only"].append(heldout(h0, p, mem))
    vec["P-only"].append(per_bug(held, h0, deploy(p), mem, EVAL_SEEDS).tolist())

    # ---------------- H-only
    rng = random.Random(seed)
    mem, h = Memory(), h0
    curve = []
    for c in range(CYCLES):
        h, *_ = h_step(p0, h, train, mem, rng)
        curve.append(heldout(h, p0, mem).fix_rate)
    curves["H-only"].append(curve)
    finals["H-only"].append(heldout(h, p0, mem))
    vec["H-only"].append(per_bug(held, h, deploy(p0), mem, EVAL_SEEDS).tolist())
    print(f"   seed {seed}: JOINT {100*finals['JOINT'][-1].fix_rate:.1f}%  "
          f"P-only {100*finals['P-only'][-1].fix_rate:.1f}%  "
          f"H-only {100*finals['H-only'][-1].fix_rate:.1f}%  (final harness JOINT [{details[('JOINT', seed)][0]}])")

# ---------------------------------------------------------------------------
banner("RESULT - held-out bugs (unseen functions), mean over training seeds")
print(f"{'arm':<10}{'fixed':>9}{'  sd across training seeds':>28}")
print(f"{'start':<10}{100*base.fix_rate:>8.1f}%{'':>28}")
for arm in ("P-only", "H-only", "JOINT"):
    v = [r.fix_rate for r in finals[arm]]
    print(f"{arm:<10}{100*st.mean(v):>8.1f}%{100*st.pstdev(v):>27.1f}")
gains = {a: st.mean([r.fix_rate for r in finals[a]]) - base.fix_rate for a in finals}
print(f"\ngain over start:  P-only {100*gains['P-only']:+.1f} pts   H-only {100*gains['H-only']:+.1f} pts   "
      f"JOINT {100*gains['JOINT']:+.1f} pts")
print(f"sum of the two single-component gains: {100*(gains['P-only']+gains['H-only']):+.1f} pts")
syn = gains["JOINT"] - (gains["P-only"] + gains["H-only"])
print(f"JOINT minus that sum (the paper's 'synergy' test): {100*syn:+.1f} pts")
print("-> positive = joint gain exceeds the sum of parts (synergy); negative = it does not.")

if details:
    seed0 = TRAIN_SEEDS[0]
    h, p = details[("JOINT", seed0)]
    print(f"\nLearned policy weights (JOINT, seed {seed0}):")
    for n_, w in sorted(zip(FEATURE_NAMES, p.w), key=lambda t: -abs(t[1]))[:5]:
        print(f"   {n_:<14}{w:+.2f}")

# figure
plt.figure(figsize=(6.4, 4))
xs = np.arange(1, CYCLES + 1)
for arm, col in (("P-only", "#c0392b"), ("H-only", "#2980b9"), ("JOINT", "#27ae60")):
    arr = 100 * np.array(curves[arm])
    plt.plot(xs, arr.mean(0), marker="o", label=arm, color=col)
    if len(arr) > 1:
        plt.fill_between(xs, arr.min(0), arr.max(0), alpha=0.15, color=col)
plt.axhline(100 * base.fix_rate, ls="--", color="grey", label="start")
plt.xlabel("Self-HPO cycle")
plt.ylabel("held-out bugs fixed (%)")
plt.xticks(xs)
plt.title("Harness-policy alternation vs single-component tuning")
plt.legend()
plt.tight_layout()
plt.savefig("outputs/part3_selfhpo_curve.png", dpi=130)
print("\nsaved outputs/part3_selfhpo_curve.png")
if not QUICK:
    vec["start"] = per_bug(held, h0, deploy(p0), None, EVAL_SEEDS).tolist()
    json.dump(vec, open("outputs/part3_results.json", "w"))
    print("saved outputs/part3_results.json (per-bug results, used by Part 4)")
print("\nCAVEATS: the H-step 'policy analysis' is rule-based, not an LLM; harness knobs are a")
print("small discrete space; the P-step also warms the SO-RSI inquiry memory (H-only has none);")
print("P-only was deployed in its trained (sampling) mode; the joint arm")
print("does twice the work per cycle (an H-step AND a P-step), so it is matched on cycles, not compute.")
