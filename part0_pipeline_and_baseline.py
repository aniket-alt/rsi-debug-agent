"""
PART 0 - The problem, the CI pipeline, and the NAIVE debugging agent (baseline).

Problem area: a debugging agent that reads failing CI tests, root-causes the
failure, patches the code and re-runs the tests.  RSI will improve that agent.

Run:  python part0_pipeline_and_baseline.py
"""
import random

from rsi_debug.agent import BUDGET, Harness, Policy, run_episode
from rsi_debug.corpus import (BASE_FUNCS, candidates, format_report, norm,
                              run_tests)
from rsi_debug.evaluate import evaluate, load_bugs


def banner(t):
    print("\n" + "=" * 78 + f"\n{t}\n" + "=" * 78)


train, held = load_bugs()

banner("0.1  A correct program, and a bug injected into it")
bug = next(b for b in held if b.func == "abs_diff_sum" and b.n_faults == 1)
print("CORRECT SOURCE:\n" + bug.correct_src)
print("\nBUGGY SOURCE (what lands in the pipeline):\n" + bug.buggy_src)
print("\n(ground truth, hidden from the agent) injected fault(s):", [str(f) for f in bug.faults])

banner("0.2  What the CI pipeline reports")
res = run_tests(bug.buggy_src, bug.visible)
print(format_report(bug.func, bug.visible, res))
print("\nThe agent sees ONLY this report + the buggy code.  It is scored on")
print(f"{len(bug.hidden)} HIDDEN tests it never sees (so over-fitting the visible tests is punished).")

banner("0.3  The agent's patch vocabulary")
print("A patch = (operator, site).  Operators:", "CMP  ARITH  EQ  BOOL  CONST_UP  CONST_DN")
cands = candidates(bug.buggy_src)
print(f"This program has {len(cands)} candidate patches:")
print("  " + ", ".join(str(c) for c in cands))
print(f"Budget: {BUDGET} test executions per bug (each CI run of the visible suite costs"
      f" {len(bug.visible)}).")

banner("0.4  Benchmark")
print(f"training bugs (9 functions): {len(train)}   held-out bugs (5 NEW functions): {len(held)}")
print("single-fault bugs:", sum(b.n_faults == 1 for b in train + held),
      "  double-fault bugs:", sum(b.n_faults == 2 for b in train + held))
print("The agent is trained/tuned on the training functions and judged on the held-out ones.")

banner("0.5  The NAIVE first-order agent, on one bug (trace)")
ep = run_episode(bug, Harness(), Policy(), random.Random(3))
for line in ep.trace:
    print("  " + line)
print(f"  -> CI green: {ep.ci_green}   hidden tests pass (truly fixed): {ep.fixed}   cost: {ep.cost}")
print(f"  -> recurrences {ep.recurrences}, opposing edits {ep.opposing}, mismatches {ep.mismatches}")

banner("0.6  Baseline on all HELD-OUT bugs (10 seeds)")
print("The naive agent tries a random patch, keeps it if CI does not get worse, no memory.")
for k in (1, 2, 3):
    for acc in ("ge", "gt"):
        print(evaluate(f"naive k={k} accept={acc}", held, Harness(k=k, accept=acc), Policy()).row())
print("\nThis is the number every RSI method below has to beat, at the SAME budget.")
