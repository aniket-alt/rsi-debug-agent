"""
self_hpo.py - Self-HPO (arXiv:2610.22791) adapted to the debugging agent.

Paper idea: harness and policy are mutually reinforcing; evolving only one is
sub-optimal.  Alternate
    H-step : the policy analyses ITS OWN failures and proposes new harnesses;
             the best one on the hard-task set is kept
    P-step : the policy is updated (RL) under the selected harness
so the harness search keeps targeting the policy's *current* failure modes.

Laptop mapping (what is real and what is simplified):
    harness  = Harness dataclass (k, aug tests, localisation, SO-RSI, accept, dedupe)
    policy   = linear patch scorer, updated with REINFORCE (a real policy gradient)
    H-step   = the failure analysis is a rule-based diagnosis over the policy's own
               episode logs (the paper's LLM proposer is NOT reproduced here)
"""
from __future__ import annotations

import random
from dataclasses import replace

import numpy as np

from .agent import Harness, Memory, Policy, run_episode
from .corpus import Bug


def reward(ep) -> float:
    return (1.0 if ep.fixed else (-0.2 if ep.ci_green else 0.0)) - 0.002 * ep.cost


# ------------------------------------------------------------------ P-step
def p_step(policy: Policy, harness: Harness, bugs, memory: Memory, rng: random.Random,
           epochs: int = 2, lr: float = 0.4):
    """REINFORCE: raise the probability of patch choices that led to a hidden-test fix."""
    pol = policy.copy()
    pol.mode = "sample"
    baseline = 0.3
    log = []
    order = list(bugs)
    for _ in range(epochs):
        rng.shuffle(order)
        for b in order:
            ep = run_episode(b, harness, pol, rng, memory=memory, learn=True)
            R = reward(ep)
            adv = R - baseline
            baseline = 0.95 * baseline + 0.05 * R
            for X, p, chosen in ep.decisions:
                for j in chosen:
                    grad = (X[j] - p @ X) / pol.temp
                    pol.w += lr * adv * grad / max(1, len(chosen)) / 10.0
            log.append(ep.fixed)
    pol.mode = "topk"
    return pol, float(np.mean(log[-len(order):]))


# ------------------------------------------------------------------ H-step
def diagnose(eps):
    n = len(eps)
    return {
        "overfit": sum(e.reason == "overfit" for e in eps) / n,
        "exhausted": sum(e.reason == "exhausted" for e in eps) / n,
        "thrash": sum(e.thrash for e in eps) / n,
        "fixed": sum(e.fixed for e in eps) / n,
    }


def propose_harnesses(h: Harness, d: dict, rng: random.Random) -> list[Harness]:
    """The policy reads its own failure statistics and proposes harness edits."""
    out = []
    if d["overfit"] > 0.05:                       # CI went green but patch was wrong
        out.append(replace(h, aug=min(10, h.aug + 3)))          # -> generate more tests
    if d["exhausted"] > 0.30:                      # ran out of budget
        out.append(replace(h, k=min(4, h.k + 1)))               # -> look at more patches
        out.append(replace(h, use_loc=True))                    # -> localise the fault
        if h.aug > 0:
            out.append(replace(h, aug=h.aug - 3))               # -> cheaper CI
    if d["thrash"] > 0.20:                         # repeating / undoing patches
        out.append(replace(h, dedupe=True))
        out.append(replace(h, so_rsi=True, accept="gt"))        # -> second-order diagnosis
    knob = rng.choice(["accept", "so_rsi", "use_loc"])          # one exploratory edit
    out.append(replace(h, accept="gt" if h.accept == "ge" else "ge") if knob == "accept"
               else replace(h, so_rsi=not h.so_rsi) if knob == "so_rsi"
               else replace(h, use_loc=not h.use_loc))
    uniq = []
    for c in out:
        if c != h and c not in uniq:
            uniq.append(c)
    return uniq


def h_step(policy: Policy, harness: Harness, bugs, memory: Memory, rng: random.Random,
           n_seeds: int = 3):
    pol = policy.copy()
    pol.mode = "topk"

    def run(h, subset):
        eps = []
        for s in range(n_seeds):
            r = random.Random(7000 + s)
            eps += [run_episode(b, h, pol, r, memory=memory.copy()) for b in subset]
        return eps

    base_eps = run(harness, bugs)
    d = diagnose(base_eps)
    # hard-task set = bugs the current harness+policy fails on most seeds
    by_bug = {}
    for e in base_eps:
        by_bug.setdefault(e.bug_id, []).append(e.fixed)
    hard = [b for b in bugs if np.mean(by_bug[b.id]) < 0.5]
    easy = [b for b in bugs if b not in hard]
    subset = hard + rng.sample(easy, min(len(easy), 20))
    best, best_score = harness, np.mean([e.fixed for e in run(harness, subset)])
    tried = []
    for cand in propose_harnesses(harness, d, rng):
        sc = np.mean([e.fixed for e in run(cand, subset)])
        tried.append((cand, sc))
        if sc > best_score + 0.01:
            best, best_score = cand, sc
    return best, d, len(hard), tried
