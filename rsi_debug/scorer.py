"""
scorer.py - Budget RSI (arXiv:2610.29121) adapted to patch selection.

Idea: do NOT run the CI suite on every candidate patch.  Instead
  1. collect a small set of labelled edits  (patch features -> CI pass rate),
  2. fit a LINEAR READOUT  score = w . [parent_rate, patch features],
  3. score every candidate for free and run CI only on the top-k.

Mapping from the paper:
    steering split  -> the visible CI tests (cheap, available to the agent)
    held-out        -> hidden tests (final scoring only)
    diff features   -> op one-hots, Ochiai suspiciousness, relative line
    96 labelled edits fit the readout (we use the same 96)
"""
from __future__ import annotations

import random

import numpy as np

from .agent import N_FEAT, Policy, featurize, ochiai
from .corpus import Bug, apply_edit, candidates, pass_rate, run_tests


def collect_labeled_edits(bugs: list[Bug], n: int = 96, seed: int = 0):
    """Evaluate n random patches (on the visible tests) -> (features, new pass rate)."""
    rng = random.Random(seed)
    X, y, cost = [], [], 0
    for _ in range(n):
        b = rng.choice(bugs)
        res = run_tests(b.buggy_src, b.visible)
        cands = candidates(b.buggy_src)
        e = rng.choice(cands)
        feats = featurize([e], pass_rate(res), ochiai(res), b.buggy_src.count("\n") + 1)[0]
        new = run_tests(apply_edit(b.buggy_src, e), b.visible)
        cost += len(b.visible) * 2           # parent run + child run
        X.append(feats)
        y.append(pass_rate(new) - pass_rate(res))   # label: CI gain of the patch
    return np.array(X), np.array(y), cost


def fit_readout(X, y, cols=None, lam: float = 1e-2) -> Policy:
    """Ridge regression.  `cols` selects which feature coordinates the readout may use."""
    cols = list(range(N_FEAT)) if cols is None else list(cols)
    A = X[:, cols]
    w_c = np.linalg.solve(A.T @ A + lam * np.eye(len(cols)), A.T @ y)
    w = np.zeros(N_FEAT)
    w[cols] = w_c
    return Policy(w=w, mode="topk")
