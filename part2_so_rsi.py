"""
PART 2 - Second-Order RSI (SO-RSI, arXiv:2610.05701)       course labs 10.06, 10.07, 10.08

   10.06  Run it      : naive (first-order) vs SO-RSI, with the dedupe ablation
   10.07  Probes      : the three anomaly types, the probe table, what the memory learned
   10.08  Apply       : replay ONE real failing bug through naive and SO-RSI

First-order agent  = patch the symptom, no memory, no questions asked.
Second-order agent = before patching, ask WHY the last attempts failed:
    recurrence  -> same patch re-tried          -> block it, probe the symptom
    opposing    -> patch undoes the last accept -> block it (oscillation)
    mismatch    -> predicted help, CI got worse -> record 'this fix class does NOT cure
                                                   this symptom' in the inquiry memory

Run:  python part2_so_rsi.py
"""
import random
import statistics as st
from collections import defaultdict

from rsi_debug.agent import (HYPOTHESIS, OPS, Harness, Memory, Policy,
                             run_episode, symptom)
from rsi_debug.corpus import format_report, run_tests
from rsi_debug.evaluate import evaluate, load_bugs


def banner(t):
    print("\n" + "=" * 78 + f"\n{t}\n" + "=" * 78)


train, held = load_bugs()
SEEDS = range(10)

# ---------------------------------------------------------------------------
banner("10.06  Run SO-RSI: naive vs second-order, same budget (40 test runs/bug)")
print("Inquiry memory is warmed up on the TRAINING bugs only (3 passes), then FROZEN,")
print("so nothing about the held-out functions leaks into it.\n")

memory = Memory()
for p in range(3):
    for b in train:
        run_episode(b, Harness(so_rsi=True), Policy(), random.Random(p), memory=memory, learn=True)
n_obs = sum(t for d in memory.stats.values() for _, t in d.values())
print(f"memory after warm-up: {len(memory.stats)} symptom classes, {n_obs} recorded observations\n")

naive = Harness()
naive_dd = Harness(dedupe=True)
so = Harness(so_rsi=True)
res = {}
res["naive"] = evaluate("NAIVE first-order", held, naive, Policy(), seeds=SEEDS)
res["dedupe"] = evaluate("naive + dedupe (ablation)", held, naive_dd, Policy(), seeds=SEEDS)
res["cold"] = evaluate("SO-RSI, empty memory", held, so, Policy(), memory=None, seeds=SEEDS)
res["warm"] = evaluate("SO-RSI, warmed memory", held, so, Policy(), memory=memory, seeds=SEEDS)
for r in res.values():
    print(r.row())
print("\nHow to read this:")
print("  * 'thrash/bug' = recurrences + opposing edits actually executed (wasted CI runs).")
print("  * 'prevented'  = proposals SO-RSI blocked before spending CI on them.")
print("  * 'naive + dedupe' isolates the cheap part (never retry); the gap to SO-RSI is the")
print("    diagnostic part (symptom probe + opposing-edit block + inquiry memory).")

# ---------------------------------------------------------------------------
banner("10.07  The three anomaly types and the diagnostic probe")
print("Anomaly            detection rule (on the patch trace)                 response")
print("-----------------  --------------------------------------------------  ---------------------------")
print("recurrence         same (patch, program state) evaluated again          block; probe symptom")
print("opposing edits     proposed patch = inverse of last accepted patch      block; avoid oscillation")
print("expectation mismatch  a proposed patch makes CI worse                   counted + negative evidence")
print("(Every tried patch writes (symptom, operator, improved?) into the inquiry memory;")
print(" mismatches are the negative evidence that lowers an operator's score for that symptom.)")

print("\nThe probe reads the CI failure and names a symptom class (cost: 2 test runs):")
print("  exc       any test raised an exception          -> suspect", sorted(HYPOTHESIS["exc"]))
print("  off1      every wrong output is off by exactly 1 -> suspect", sorted(HYPOTHESIS["off1"]))
print("  signflip  every wrong output = -expected         -> suspect", sorted(HYPOTHESIS["signflip"]))
print("  sparse    <=40% of tests fail                    -> suspect", sorted(HYPOTHESIS["sparse"]))
print("  dense     most tests fail                        -> suspect", sorted(HYPOTHESIS["dense"]))
print("These starting hypotheses are human-authored heuristics (the paper's 'lightweight probes').")

print("\nWhat did the inquiry memory LEARN?  P(operator fixes | symptom), top 3 per symptom:")
print(f"  {'symptom':<10}{'operator':<11}{'learned':>8}{'trials':>8}   human hypothesis?")
for sig in ("exc", "off1", "signflip", "sparse", "dense"):
    if sig not in memory.stats:
        continue
    rows = sorted(((memory.success_rate(sig, o), memory.stats[sig][o][1], o) for o in OPS
                   if memory.stats[sig][o][1] > 0), reverse=True)[:3]
    for rate, trials, op in rows:
        print(f"  {sig:<10}{op:<11}{rate:>8.2f}{trials:>8}   {'agrees' if op in HYPOTHESIS[sig] else 'DISAGREES'}")
dis = sum(1 for sig in memory.stats for o in OPS if memory.stats[sig][o][1] > 2
          and o not in HYPOTHESIS.get(sig, ()) and memory.success_rate(sig, o) > 0.3)
print(f"\n  -> {dis} pairs with >2 trials and learned rate >0.3 that the human table did NOT predict.")
print("     Inquiry memory lets observed evidence override the initial guess - but note the")
print("     small-sample risk: 'off1/EQ' = 0.66 rests on only 13 trials.")

print("\nAnomaly frequency in NAIVE traces on held-out bugs (10 seeds):")
cnt = defaultdict(list)
for s in SEEDS:
    rng = random.Random(1000 + s)
    for b in held:
        e = run_episode(b, naive, Policy(), rng)
        cnt["recurrence"].append(e.recurrences)
        cnt["opposing"].append(e.opposing)
        cnt["mismatch"].append(e.mismatches)
for k, v in cnt.items():
    print(f"  {k:<11} {st.mean(v):.2f} per bug   ({100 * sum(x > 0 for x in v) / len(v):.0f}% of bugs hit it)")

# ---------------------------------------------------------------------------
banner("10.08  Apply it: one real failing bug, replayed twice")
# find a held-out bug that the naive agent fails (with anomalies) and SO-RSI fixes
pick = None
for b in held:
    for s in range(30):
        en = run_episode(b, naive, Policy(), random.Random(s))
        if en.fixed or (en.thrash + en.mismatches) == 0:
            continue
        es = run_episode(b, so, Policy(), random.Random(s), memory=memory.copy())
        if es.fixed:
            pick = (b, s, en, es)
            break
    if pick:
        break
b, s, en, es = pick
print("BUGGY CODE:\n" + b.buggy_src)
print("\nCI REPORT:\n" + format_report(b.func, b.visible, run_tests(b.buggy_src, b.visible)))
print(f"\n--- A) NAIVE first-order agent (seed {s}) ---")
for line in en.trace:
    print("  " + line)
print(f"  => truly fixed: {en.fixed}   wasted: {en.recurrences} recurrences, {en.opposing} opposing, "
      f"{en.mismatches} mismatches")
print(f"\n--- B) SO-RSI agent (same bug, same seed) ---")
for line in es.trace:
    print("  " + line)
print(f"  => truly fixed: {es.fixed}   cost: {es.cost}/40")
print("\nFIXED SOURCE (SO-RSI):\n" + es.final_src)
print("\nRetrospective analysis (what the course asks in lab 10.08):")
print("  * Did the naive agent repeat a patch?        recurrences =", en.recurrences)
print("  * Did it undo a previous change?             opposing    =", en.opposing)
print("  * Did an outcome surprise it?                mismatches  =", en.mismatches)
print("  * What SO-RSI did differently: it probed the symptom, blocked every repeat, and never")
print("    ran the oscillating CONST_UP/CONST_DN pair.  It did NOT get the ranking right")
print("    at first: the memory's learned bonus for EQ (from few trials, see 10.07) outranked")
print("    the probe's CMP/CONST suspects, so rounds 1-3 were wasted on EQ patches.  The")
print("    blocking logic kept it from repeating them, and round 4 found the fix.")
print("\nNOTE: this is one hand-picked illustrative bug (selected because it shows the")
print("mechanism, including a weakness).  The honest numbers are the 10-seed table in 10.06.")
