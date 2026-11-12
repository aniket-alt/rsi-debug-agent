"""
corpus.py - a miniature CI pipeline with injected bugs.

The problem area for this assignment: a DEBUGGING AGENT that reads failing CI
tests, root-causes them, patches the code and re-runs the tests.

Everything here is real executable Python (no LLM, no mocks):
  * BASE_FUNCS    - 14 small, correct functions
  * edit operators - mutation operators used BOTH to inject bugs and as the
                    agent's patch vocabulary (a patch is an (op, site) pair)
  * run_tests     - the "CI runner": executes tests, records pass/fail, the
                    error, and which source lines each test touched
  * make_bugs     - builds a bug benchmark (single and double faults) with
                    three test sets: visible (what CI shows the agent),
                    aug (extra generated tests the harness may add) and
                    hidden (ground truth used only to score the final patch)
"""
from __future__ import annotations

import ast
import math
import random
import sys
from dataclasses import dataclass, field

# --------------------------------------------------------------------------
# 1. Correct programs + input generators
# --------------------------------------------------------------------------

def _ints(r, n, lo=-3, hi=3):
    return [r.randint(lo, hi) for _ in range(n)]


def _g_bsum(r, b):
    xs = _ints(r, r.randint(0, 6), -3, 6)
    cap = r.randint(1, 10)
    if b:
        cap = max(1, sum(x for x in xs if x >= 0))
    return (xs, cap)


def _g_sum_to_n(r, b):
    return (r.choice([0, 1, 2]) if b else r.randint(0, 10),)


def _g_count_pos(r, b):
    xs = _ints(r, r.randint(0, 6))
    if b:
        xs += [0, 1]
        r.shuffle(xs)
    return (xs,)


def _g_max_of(r, b):
    xs = _ints(r, r.randint(1, 6), -5, 5)
    if b:
        m = max(xs) + 1
        xs.insert(r.choice([0, len(xs)]), m)
    return (xs,)


def _g_fizz(r, b):
    return (r.choice([3, 5, 15, 30, 9, 10]) if b else r.randint(1, 30),)


def _g_fact(r, b):
    return (r.choice([0, 1, 2]) if b else r.randint(0, 7),)


def _g_inv(r, b):
    xs = _ints(r, r.randint(0, 6))
    if b and xs:
        xs.sort(reverse=r.random() < 0.5)
    return (xs,)


def _g_lsearch(r, b):
    xs = _ints(r, r.randint(0, 6), 0, 6)
    if b and xs:
        return (xs, xs[-1])
    return (xs, r.randint(0, 6))


def _g_avg(r, b):
    xs = _ints(r, r.randint(1, 6), -4, 8)
    if b:
        xs = [-1] * r.randint(1, 3)
    return (xs,)


def _g_inrange(r, b):
    lo, hi = r.randint(-3, 0), r.randint(1, 4)
    xs = _ints(r, r.randint(0, 6), -5, 6)
    if b:
        xs += [lo, hi]
    return (xs, lo, hi)


def _g_fib(r, b):
    return (r.choice([0, 1, 2]) if b else r.randint(0, 12),)


def _g_gcd(r, b):
    a, c = r.randint(1, 30), r.randint(1, 30)
    if b:
        c = a * r.choice([1, 2])
    return (a, c)


def _g_absdiff(r, b):
    xs = _ints(r, r.randint(0, 6), -5, 5)
    if b:
        xs = sorted(xs, reverse=r.random() < 0.5)
    return (xs,)


def _g_leap(r, b):
    return (r.choice([1900, 2000, 2100, 2024, 2023, 1600]) if b else r.randint(1500, 2200),)


BASE_FUNCS = {
    "bounded_sum": ("""
def bounded_sum(xs, cap):
    t = 0
    for x in xs:
        if x < 0:
            continue
        t += x
        if t > cap:
            return cap
    return t
""", _g_bsum),
    "sum_to_n": ("""
def sum_to_n(n):
    total = 0
    for i in range(1, n + 1):
        total += i
    return total
""", _g_sum_to_n),
    "count_positive": ("""
def count_positive(xs):
    c = 0
    for x in xs:
        if x > 0:
            c += 1
    return c
""", _g_count_pos),
    "max_of": ("""
def max_of(xs):
    m = xs[0]
    for x in xs[1:]:
        if x > m:
            m = x
    return m
""", _g_max_of),
    "factorial": ("""
def factorial(n):
    r = 1
    for i in range(2, n + 1):
        r *= i
    return r
""", _g_fact),
    "linear_search": ("""
def linear_search(xs, t):
    for i in range(len(xs)):
        if xs[i] == t:
            return i
    return -1
""", _g_lsearch),
    "avg_floor": ("""
def avg_floor(xs):
    s = 0
    for x in xs:
        s += x
    return s // len(xs)
""", _g_avg),
    "in_range_count": ("""
def in_range_count(xs, lo, hi):
    c = 0
    for x in xs:
        if x >= lo and x <= hi:
            c += 1
    return c
""", _g_inrange),
    "fib": ("""
def fib(n):
    a, b = 0, 1
    for _ in range(n):
        a, b = b, a + b
    return a
""", _g_fib),
    # ---- held-out functions (never seen while training the agent) ----
    "fizz": ("""
def fizz(n):
    if n % 15 == 0:
        return 3
    if n % 5 == 0:
        return 2
    if n % 3 == 0:
        return 1
    return 0
""", _g_fizz),
    "inversions": ("""
def inversions(xs):
    c = 0
    for i in range(len(xs)):
        for j in range(i + 1, len(xs)):
            if xs[i] > xs[j]:
                c += 1
    return c
""", _g_inv),
    "gcd": ("""
def gcd(a, b):
    while b != 0:
        a, b = b, a % b
    return a
""", _g_gcd),
    "abs_diff_sum": ("""
def abs_diff_sum(xs):
    t = 0
    for i in range(len(xs) - 1):
        d = xs[i] - xs[i + 1]
        if d < 0:
            d = -d
        t += d
    return t
""", _g_absdiff),
    "leap": ("""
def leap(y):
    if y % 4 == 0 and (y % 100 != 0 or y % 400 == 0):
        return 1
    return 0
""", _g_leap),
}

TRAIN_FUNCS = ["bounded_sum", "sum_to_n", "count_positive", "max_of", "factorial",
               "linear_search", "avg_floor", "in_range_count", "fib"]
HELDOUT_FUNCS = ["fizz", "inversions", "gcd", "abs_diff_sum", "leap"]


def norm(src: str) -> str:
    """Canonical formatting so line numbers are stable across edits."""
    return ast.unparse(ast.parse(src.strip() + "\n"))


# --------------------------------------------------------------------------
# 2. Edit operators (patch vocabulary)
# --------------------------------------------------------------------------
OPS = ["CMP", "ARITH", "EQ", "BOOL", "CONST_UP", "CONST_DN"]
INVERSE = {"CMP": "CMP", "ARITH": "ARITH", "EQ": "EQ", "BOOL": "BOOL",
           "CONST_UP": "CONST_DN", "CONST_DN": "CONST_UP"}

_CMP = {ast.Lt: ast.LtE, ast.LtE: ast.Lt, ast.Gt: ast.GtE, ast.GtE: ast.Gt}
_ARITH = {ast.Add: ast.Sub, ast.Sub: ast.Add, ast.Mult: ast.FloorDiv, ast.FloorDiv: ast.Mult}
_EQ = {ast.Eq: ast.NotEq, ast.NotEq: ast.Eq}
_BOOL = {ast.And: ast.Or, ast.Or: ast.And}


@dataclass(frozen=True)
class Edit:
    op: str
    site: int          # index among nodes applicable to `op`
    line: int          # source line of the node (for fault localisation)

    def key(self):
        return (self.op, self.site)

    def inverse(self):
        return Edit(INVERSE[self.op], self.site, self.line)

    def __str__(self):
        return f"{self.op}@L{self.line}#{self.site}"


def _sites(tree, op):
    out = []
    for n in ast.walk(tree):
        if op == "CMP" and isinstance(n, ast.Compare) and len(n.ops) == 1 and type(n.ops[0]) in _CMP:
            out.append(n)
        elif op == "EQ" and isinstance(n, ast.Compare) and len(n.ops) == 1 and type(n.ops[0]) in _EQ:
            out.append(n)
        elif op == "ARITH" and ((isinstance(n, ast.BinOp) and type(n.op) in _ARITH)
                                or (isinstance(n, ast.AugAssign) and type(n.op) in _ARITH)):
            out.append(n)
        elif op == "BOOL" and isinstance(n, ast.BoolOp):
            out.append(n)
        elif op in ("CONST_UP", "CONST_DN") and isinstance(n, ast.Constant) \
                and isinstance(n.value, int) and not isinstance(n.value, bool):
            out.append(n)
    return out


def candidates(src: str) -> list[Edit]:
    """Every patch the agent could apply to `src`."""
    tree = ast.parse(src)
    edits = []
    for op in OPS:
        for i, n in enumerate(_sites(tree, op)):
            edits.append(Edit(op, i, getattr(n, "lineno", 0)))
    return edits


def apply_edit(src: str, e: Edit) -> str:
    tree = ast.parse(src)
    n = _sites(tree, e.op)[e.site]
    if e.op == "CMP":
        n.ops[0] = _CMP[type(n.ops[0])]()
    elif e.op == "EQ":
        n.ops[0] = _EQ[type(n.ops[0])]()
    elif e.op == "ARITH":
        n.op = _ARITH[type(n.op)]()
    elif e.op == "BOOL":
        n.op = _BOOL[type(n.op)]()
    elif e.op == "CONST_UP":
        n.value += 1
    elif e.op == "CONST_DN":
        n.value -= 1
    return ast.unparse(tree)


# --------------------------------------------------------------------------
# 3. The CI runner
# --------------------------------------------------------------------------
LINE_LIMIT = 4000


class _Timeout(Exception):
    pass


@dataclass
class TestResult:
    passed: bool
    actual: object = None
    expected: object = None
    error: str | None = None
    lines: frozenset = frozenset()


def _fname(src):
    return ast.parse(src).body[0].name


def run_tests(src: str, tests) -> list[TestResult]:
    """Run every (args, expected) test against `src`. This is one 'CI run'."""
    ns: dict = {}
    try:
        exec(compile(src, "<prog>", "exec"), ns)
        fn = ns[_fname(src)]
    except Exception as ex:  # a patch that does not even compile
        return [TestResult(False, error=f"{type(ex).__name__}: {ex}") for _ in tests]
    results = []
    for args, expected in tests:
        seen, count = set(), [0]

        def tracer(frame, event, arg):
            if frame.f_code.co_filename != "<prog>":
                return None
            if event == "line":
                seen.add(frame.f_lineno)
                count[0] += 1
                if count[0] > LINE_LIMIT:
                    raise _Timeout()
            return tracer

        old = sys.gettrace()
        sys.settrace(tracer)
        try:
            actual = fn(*[list(a) if isinstance(a, list) else a for a in args])
            ok = actual == expected
            results.append(TestResult(ok, actual, expected, None, frozenset(seen)))
        except _Timeout:
            results.append(TestResult(False, None, expected, "Timeout: infinite loop", frozenset(seen)))
        except Exception as ex:
            results.append(TestResult(False, None, expected, f"{type(ex).__name__}: {ex}", frozenset(seen)))
        finally:
            sys.settrace(old)
    return results


def pass_rate(results) -> float:
    return sum(r.passed for r in results) / max(1, len(results))


def format_report(name, tests, results, limit=3) -> str:
    """What the CI pipeline prints for a developer (or the agent) to read."""
    bad = [(t, r) for t, r in zip(tests, results) if not r.passed]
    lines = [f"CI: {len(results) - len(bad)}/{len(results)} tests passed"]
    for (args, exp), r in bad[:limit]:
        got = r.error if r.error else repr(r.actual)
        lines.append(f"  FAIL {name}{args!r}: expected {exp!r}, got {got}")
    if len(bad) > limit:
        lines.append(f"  ... and {len(bad) - limit} more failures")
    return "\n".join(lines)


# --------------------------------------------------------------------------
# 4. Bug benchmark
# --------------------------------------------------------------------------
@dataclass
class Bug:
    id: str
    func: str
    correct_src: str
    buggy_src: str
    faults: list                # injected Edits (ground truth, never shown to agent)
    visible: list               # tests the CI shows the agent
    aug: list                   # extra generated tests the harness may enable
    hidden: list                # ground-truth tests, used only for final scoring

    @property
    def n_faults(self):
        return len(self.faults)


def _make_tests(gen, correct_src, rng, n, boundary_p):
    ns: dict = {}
    exec(compile(correct_src, "<correct>", "exec"), ns)
    fn = ns[_fname(correct_src)]
    tests = []
    for _ in range(n):
        args = gen(rng, rng.random() < boundary_p)
        tests.append((args, fn(*[list(a) if isinstance(a, list) else a for a in args])))
    return tests


def make_bugs(funcs, seed=0, singles=6, doubles=4, n_visible=5, n_aug=10, n_hidden=80):
    rng = random.Random(seed)
    bugs = []
    for name in funcs:
        raw, gen = BASE_FUNCS[name]
        correct = norm(raw)
        hidden = _make_tests(gen, correct, random.Random(f"h-{name}"), n_hidden, 0.5)
        cands = candidates(correct)
        rng.shuffle(cands)
        made = 0
        # ---- single-fault bugs
        for e in cands:
            if made >= singles:
                break
            bsrc = apply_edit(correct, e)
            bug = _finish(name, correct, bsrc, [e], gen, hidden, rng, n_visible, n_aug, f"{name}-s{made}")
            if bug:
                bugs.append(bug)
                made += 1
        # ---- double-fault bugs (two independent edits, different lines)
        made = 0
        tries = 0
        while made < doubles and tries < 200:
            tries += 1
            e1, e2 = rng.sample(cands, 2)
            if e1.line == e2.line:
                continue
            s1 = apply_edit(correct, e1)
            c2 = [c for c in candidates(s1) if c.key() == e2.key()]
            if not c2:
                continue
            bsrc = apply_edit(s1, c2[0])
            bug = _finish(name, correct, bsrc, [e1, c2[0]], gen, hidden, rng, n_visible, n_aug, f"{name}-d{made}")
            if bug:
                bugs.append(bug)
                made += 1
    return bugs


def _finish(name, correct, bsrc, faults, gen, hidden, rng, n_visible, n_aug, bid):
    if run_tests(bsrc, hidden) and pass_rate(run_tests(bsrc, hidden)) == 1.0:
        return None                          # equivalent mutant: not a real bug
    for _ in range(30):
        visible = _make_tests(gen, correct, random.Random(rng.random()), n_visible, 0.2)
        if pass_rate(run_tests(bsrc, visible)) < 1.0:   # CI must notice the bug
            aug = _make_tests(gen, correct, random.Random(rng.random()), n_aug, 0.7)
            return Bug(bid, name, correct, bsrc, faults, visible, aug, hidden)
    return None


def is_fixed(bug: Bug, src: str) -> bool:
    """Ground truth: does the patched program pass the HIDDEN test set?"""
    return pass_rate(run_tests(src, bug.hidden)) == 1.0
