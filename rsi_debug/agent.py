"""
agent.py - the debugging agent and its three improvable parts.

The agent loop (one 'episode' = one bug):
    run CI -> read failures -> propose patches -> re-run CI -> keep or discard
until all CI tests pass or the test-execution BUDGET is spent.

Three objects the RSI methods will change (see Theme 09 "three objects"):
    Policy   - weights that score candidate patches            (Budget RSI, Self-HPO P-step)
    Harness  - how the loop is run: k, extra tests, localisation, diagnostics
                                                               (Self-HPO H-step)
    Memory   - SO-RSI inquiry memory: causal evidence about which fix class
               cures which symptom                              (SO-RSI)
"""
from __future__ import annotations

import hashlib
import math
import random
from collections import Counter, defaultdict
from dataclasses import dataclass, field, replace

import numpy as np

from .corpus import (OPS, INVERSE, Bug, Edit, apply_edit, candidates, is_fixed,
                     pass_rate, run_tests)

BUDGET = 40          # test executions the agent may spend per bug (all methods)
N_FEAT = 9           # parent_rate + 6 op one-hots + ochiai + relative line
FEATURE_NAMES = ["parent_rate"] + [f"op={o}" for o in OPS] + ["ochiai", "rel_line"]


# --------------------------------------------------------------------------
# Harness / Policy / Memory
# --------------------------------------------------------------------------
@dataclass(frozen=True)
class Harness:
    k: int = 1               # patches fully evaluated per round
    aug: int = 0             # extra generated tests added to the CI suite
    use_loc: bool = False    # fault localisation (Ochiai) feature available
    so_rsi: bool = False     # SO-RSI diagnostics + inquiry memory on
    accept: str = "ge"       # 'ge' keep patch if tests no worse; 'gt' only if strictly better
    dedupe: bool = False     # never re-evaluate a patch already tried on this exact program state

    def __str__(self):
        return (f"k={self.k} aug={self.aug} loc={int(self.use_loc)} so_rsi={int(self.so_rsi)} "
                f"accept={self.accept} dedupe={int(self.dedupe)}")


@dataclass
class Policy:
    """Linear scorer over patch features.  score = w . phi(patch)."""
    w: np.ndarray = field(default_factory=lambda: np.zeros(N_FEAT))
    mode: str = "topk"        # 'topk' (deploy) or 'sample' (softmax, used for RL)
    temp: float = 0.15

    def copy(self):
        return Policy(self.w.copy(), self.mode, self.temp)


class Memory:
    """SO-RSI inquiry memory: persistent causal evidence across episodes.

    stats[symptom][op] = [successes, trials]  — did op X cure symptom Y?
    """

    def __init__(self):
        self.stats = defaultdict(lambda: defaultdict(lambda: [0, 0]))

    def record(self, sig, op, improved):
        s = self.stats[sig][op]
        s[1] += 1
        s[0] += int(improved)

    def success_rate(self, sig, op):
        s, t = self.stats[sig][op]
        return (s + 0.2) / (t + 1.0)

    def copy(self):
        m = Memory()
        for sig, d in self.stats.items():
            for op, (s, t) in d.items():
                m.stats[sig][op] = [s, t]
        return m


# --------------------------------------------------------------------------
# Features, fault localisation, symptoms
# --------------------------------------------------------------------------
def ochiai(tests_results):
    """Spectrum-based fault localisation: suspiciousness per source line."""
    fails = [r for r in tests_results if not r.passed]
    tot_f = len(fails)
    if tot_f == 0:
        return {}
    lines = set().union(*[r.lines for r in tests_results])
    out = {}
    for ln in lines:
        ef = sum(ln in r.lines for r in fails)
        ep = sum(ln in r.lines for r in tests_results if r.passed)
        den = math.sqrt(tot_f * (ef + ep))
        out[ln] = ef / den if den else 0.0
    return out


def featurize(cands: list[Edit], parent_rate: float, susp: dict, n_lines: int) -> np.ndarray:
    X = np.zeros((len(cands), N_FEAT))
    for i, e in enumerate(cands):
        X[i, 0] = parent_rate
        X[i, 1 + OPS.index(e.op)] = 1.0
        X[i, 7] = susp.get(e.line, 0.0)
        X[i, 8] = e.line / max(1, n_lines)
    return X


def symptom(tests, results) -> str:
    """Summarise the failure pattern into a coarse symptom signature (a probe)."""
    bad = [r for r in results if not r.passed]
    if not bad:
        return "ok"
    if any(r.error for r in bad):
        return "exc"
    diffs = []
    for r in bad:
        if isinstance(r.actual, int) and isinstance(r.expected, int):
            diffs.append((r.actual, r.expected))
        else:
            return "dense"
    if all(abs(a - e) == 1 for a, e in diffs):
        return "off1"
    if all(a == -e and e != 0 for a, e in diffs):
        return "signflip"
    return "sparse" if len(bad) / len(results) <= 0.4 else "dense"


# Human-authored starting hypotheses ("lightweight diagnostic probes", paper sec. 3).
HYPOTHESIS = {
    "exc": {"CONST_UP", "CONST_DN", "ARITH"},
    "off1": {"CONST_UP", "CONST_DN", "CMP"},
    "signflip": {"ARITH"},
    "sparse": {"CMP", "EQ", "BOOL"},
    "dense": {"BOOL", "EQ", "ARITH"},
}


def hypothesis_bonus(sig, op, memory: Memory | None) -> float:
    h = 1.0 if op in HYPOTHESIS.get(sig, ()) else 0.0
    learned = memory.success_rate(sig, op) if memory is not None else 0.0
    return 0.2 * h + 0.8 * learned


def _h(src):
    return hashlib.md5(src.encode()).hexdigest()[:8]


# CI results are deterministic, so identical (bug, suite, program) runs are cached.
# The agent's *logical* cost is still charged per run; this only saves wall-clock.
_CI_CACHE: dict = {}


def ci_run(bug: Bug, aug: int, src: str):
    key = (bug.id, aug, src)
    if key not in _CI_CACHE:
        _CI_CACHE[key] = run_tests(src, bug.visible + bug.aug[:aug])
    return _CI_CACHE[key]


_FIX_CACHE: dict = {}


def cached_is_fixed(bug: Bug, src: str) -> bool:
    key = (bug.id, src)
    if key not in _FIX_CACHE:
        _FIX_CACHE[key] = is_fixed(bug, src)
    return _FIX_CACHE[key]


# --------------------------------------------------------------------------
# The episode
# --------------------------------------------------------------------------
@dataclass
class Episode:
    bug_id: str
    fixed: bool                 # hidden tests all pass (ground truth)
    ci_green: bool              # the agent's own CI went green (it believed it was done)
    cost: int                   # test executions used
    rounds: int
    recurrences: int = 0        # same patch re-tried on the same program state
    opposing: int = 0           # patch that undoes the previous accepted patch
    mismatches: int = 0         # patch predicted helpful but made CI worse
    prevented: int = 0          # anomalies SO-RSI blocked instead of executing
    probes: int = 0
    trace: list = field(default_factory=list)      # human-readable log
    decisions: list = field(default_factory=list)  # (X, chosen idx) for RL
    reason: str = ""            # fixed | overfit | exhausted

    @property
    def thrash(self):
        return self.recurrences + self.opposing


def run_episode(bug: Bug, harness: Harness, policy: Policy, rng: random.Random,
                memory: Memory | None = None, learn: bool = False,
                budget: int = BUDGET, verbose: bool = False) -> Episode:
    tests = bug.visible + bug.aug[:harness.aug]
    src = bug.buggy_src
    ep = Episode(bug.id, False, False, 0, 0)
    log = ep.trace.append

    res = ci_run(bug, harness.aug, src)
    ep.cost += len(tests)
    rate = pass_rate(res)
    tried: set = set()             # (state_hash, edit_key) already evaluated
    tabu_sites: set = set()        # SO-RSI: sites blocked after an oscillation
    last_accepted: Edit | None = None
    last_sig_fail = None
    stagnant = 0
    rng_np = np.random.default_rng(rng.randrange(2 ** 32))

    while rate < 1.0 and ep.cost + len(tests) * harness.k <= budget:
        ep.rounds += 1
        cands = candidates(src)
        if not cands:
            break
        sig = symptom(tests, res)
        susp = ochiai(res) if harness.use_loc else {}
        n_lines = src.count("\n") + 1
        X = featurize(cands, rate, susp, n_lines)
        score = X @ policy.w

        # -------- SO-RSI: second-order diagnosis happens BEFORE choosing a patch
        allowed = np.ones(len(cands), bool)
        if harness.so_rsi:
            fail_sig = tuple(i for i, r in enumerate(res) if not r.passed)
            stagnant = stagnant + 1 if fail_sig == last_sig_fail else 0
            last_sig_fail = fail_sig
            n_rec = n_opp = 0
            for i, e in enumerate(cands):
                if (_h(src), e.key()) in tried:                  # recurrence blocked
                    allowed[i] = False
                    n_rec += 1
                if (e.op, e.site) in tabu_sites:                  # oscillating site blocked
                    allowed[i] = False
                if last_accepted is not None and e.key() == last_accepted.inverse().key():
                    allowed[i] = False                           # opposing edit blocked
                    n_opp += 1
            ep.prevented += n_rec + n_opp
            ep.probes += 1
            ep.cost += 2                                          # a probe costs 2 CI executions
            bonus = np.array([hypothesis_bonus(sig, e.op, memory) for e in cands])
            score = score + bonus
            top_hyp = sorted(HYPOTHESIS.get(sig, ()))
            log(f"r{ep.rounds} PROBE symptom={sig!r} -> suspect {top_hyp}; "
                f"blocked {n_rec} recurring + {n_opp} opposing proposals")
            if not allowed.any():
                break
        if harness.dedupe and not harness.so_rsi:
            for i, e in enumerate(cands):
                if (_h(src), e.key()) in tried:
                    allowed[i] = False
        idx_pool = np.flatnonzero(allowed)
        if len(idx_pool) == 0:
            break

        # -------- choose up to k patches
        k = min(harness.k, len(idx_pool))
        if policy.mode == "sample":
            z = score[idx_pool] / policy.temp
            p = np.exp(z - z.max())
            p /= p.sum()
            chosen = list(rng_np.choice(idx_pool, size=k, replace=False, p=p))
            ep.decisions.append((X[idx_pool], p, [int(np.flatnonzero(idx_pool == c)[0]) for c in chosen]))
        elif policy.w.any() or harness.so_rsi:
            noise = rng_np.random(len(idx_pool)) * 1e-6
            order = np.argsort(-(score[idx_pool] + noise))
            chosen = [idx_pool[o] for o in order[:k]]
        else:  # untrained policy: uniform random sampling WITH replacement across rounds
            chosen = list(rng_np.choice(idx_pool, size=k, replace=False))

        # -------- evaluate the chosen patches on the CI suite
        best = None
        for ci in chosen:
            e = cands[ci]
            key = (_h(src), e.key())
            if key in tried:
                ep.recurrences += 1
                log(f"r{ep.rounds} !! RECURRENCE: {e} was already tried on this exact program")
            if last_accepted is not None and e.key() == last_accepted.inverse().key():
                ep.opposing += 1
                log(f"r{ep.rounds} !! OPPOSING EDIT: {e} undoes the previously accepted {last_accepted}")
            tried.add(key)
            new_src = apply_edit(src, e)
            new_res = ci_run(bug, harness.aug, new_src)
            ep.cost += len(tests)
            new_rate = pass_rate(new_res)
            improved = new_rate > rate
            if new_rate < rate:
                ep.mismatches += 1
                log(f"r{ep.rounds} !! EXPECTATION MISMATCH: {e} was expected to help but CI fell "
                    f"{rate:.2f}->{new_rate:.2f}")
            if harness.so_rsi and memory is not None and learn:
                memory.record(sig, e.op, improved)
            log(f"r{ep.rounds} try {e}: CI {rate:.2f}->{new_rate:.2f}")
            if best is None or new_rate > best[0]:
                best = (new_rate, new_src, new_res, e)

        # -------- accept / reject
        if best is not None:
            ok = best[0] > rate if harness.accept == "gt" else best[0] >= rate
            if ok:
                src, res, rate, last_accepted = best[1], best[2], best[0], best[3]
                tabu_sites.clear()          # program changed: earlier rejections may no longer hold
                log(f"r{ep.rounds} ACCEPT {best[3]} (CI={rate:.2f})")
            elif harness.so_rsi:
                tabu_sites.add(best[3].key())

    ep.ci_green = rate >= 1.0
    ep.fixed = cached_is_fixed(bug, src)
    ep.reason = "fixed" if ep.fixed else ("overfit" if ep.ci_green else "exhausted")
    ep.final_src = src
    return ep
