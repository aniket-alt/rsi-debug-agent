# Assignment 5 - Recursive Self-Improvement for a CI Debugging Agent

**Problem area:** a debugging agent that reads failing CI tests, root-causes the failure,
patches the code and re-runs the tests - and three October-2026 RSI methods that improve it.

| Part | Course labs (Theme 10) | RSI method | Script | Saved output |
|---|---|---|---|---|
| 0 | setup | pipeline + naive baseline | `part0_pipeline_and_baseline.py` | `outputs/part0_output.txt` |
| 1 | 10.03 - 10.05 | **Budget RSI** (arXiv 2610.29121) | `part1_budget_rsi.py` | `outputs/part1_output.txt` |
| 2 | 10.06 - 10.08 | **SO-RSI** (arXiv 2610.05701) | `part2_so_rsi.py` | `outputs/part2_output.txt` |
| 3 | 10.12, 10.14 | **Self-HPO** (arXiv 2610.22791) | `part3_self_hpo.py` | `outputs/part3_output.txt` |
| 4 | Theme 09 / 11 | fair comparison + claim | `part4_final_comparison.py` | `outputs/part4_output.txt` |

Methods and lab content come from `COURSE-MAP.md` (Theme 10) of
`dlmastery/simple-coding-harness`. The assignment asks for three recent RSI methods and names
dream-rsi, AIDE2 and Scientist2 only as examples, so I chose three October 2026 papers covered by
the course's Theme 10: Budget RSI, SO-RSI and Self-HPO.

## Videos (one per part)
| Part | Video |
|---|---|
| 0 - Pipeline + naive baseline | [Video](https://youtube.com/PLACEHOLDER-part0) |
| 1 - Budget RSI (labs 10.03-10.05) | [Video](https://youtube.com/PLACEHOLDER-part1) |
| 2 - SO-RSI (labs 10.06-10.08) | [Video](https://youtube.com/PLACEHOLDER-part2) |
| 3 - Self-HPO (labs 10.12, 10.14) | [Video](https://youtube.com/PLACEHOLDER-part3) |
| 4 - Comparison + claim | [Video](https://youtube.com/PLACEHOLDER-part4) |

## Run it
```bash
pip install -r requirements.txt          # numpy, matplotlib; Python 3.10+
python part0_pipeline_and_baseline.py    # ~10 s
python part1_budget_rsi.py               # ~1 min
python part2_so_rsi.py                   # ~15 s
python part3_self_hpo.py --quick         # ~1 min   (omit --quick: 3 seeds, ~4 min; writes the JSON Part 4 needs)
python part4_final_comparison.py         # ~2 min   (needs the full Part 3 run)
```
No API key, no GPU, no network. Everything is real executable Python.

## The setup (Part 0)
* 14 tiny correct functions; bugs are 1- or 2-operator mutations (`<`->`<=`, `+`->`-`, constant +/-1,
  `and`->`or`, `==`->`!=`). 79 training bugs (9 functions), **43 held-out bugs (5 functions never used
  for fitting or tuning)**.
* `run_tests` is the CI runner. The agent sees only the visible failing tests; a **hidden** test set scores
  the final patch, so over-fitting the visible tests is punished.
* A patch is an `(operator, site)` pair. Budget: **40 test executions per bug** for every method.
* The agent loop: run CI -> propose patch(es) -> re-run CI -> keep/discard -> repeat until green or budget out.

## Headline result (held-out bugs, same budget, 95% bootstrap CI over bugs)
| method | fixed | gain vs naive |
|---|---|---|
| Naive first-order | 14.2% [9.8, 19.3] | - |
| SO-RSI (warmed memory) | 26.5% [15.6, 38.4] | +12.3 [+3.0, +21.9] |
| Self-HPO (from scratch, 3 seeds) | 38.1% [25.7, 50.4] | +23.9 [+14.4, +33.9] |
| Self-HPO + Budget-RSI init | 39.8% [28.7, 51.0] | +25.7 [+17.5, +34.0] |
| Budget RSI | 44.0% [30.7, 58.1] | +29.8 [+17.7, +41.6] |
| Budget RSI + SO-RSI | 45.8% [31.4, 60.2] | +31.6 [+19.1, +44.7] |

Figure: `outputs/part4_comparison.png`, `outputs/part3_selfhpo_curve.png`.

## What the results do and do not show (honest reading)
* Every method beats naive; the CIs of the top four overlap heavily, so **I cannot rank Budget RSI,
  Budget+SO-RSI and Self-HPO against each other.** The simplest method scored highest.
* **Budget RSI:** the linear readout's one-shot ranking edge is small (first valid fix at mean rank 7.2
  vs 7.9 random); the gain compounds over rounds. It is uneven by function (helps `abs_diff_sum`,
  `gcd`; does nothing on `leap`). The localisation feature got ~0 weight. The 96-patch fit costs
  960 test runs (~22/bug amortised), not counted in the per-bug budget.
* **SO-RSI:** "never retry a patch" alone (dedupe) gave **no** gain (13.5% vs 14.2%). The gain comes from
  the probes + inquiry memory (19.3% cold -> 26.5% warmed). The memory can mislead from small samples
  (see the 10.08 trace, where `off1 -> EQ` from 13 trials wasted three rounds).
* **Self-HPO:** the paper's claim "joint > sum of parts" was **not reproduced**
  (joint minus sum of single gains = -12.1 pts). Most of the gain is the H-step discovering that the
  SO-RSI harness should be on; the policy-gradient step adds ~0.5 pts on top of H-only.
* **Limits:** synthetic mutation bugs in tiny functions; the agent searches a fixed patch vocabulary and is
  not an LLM; the H-step "analysis" is rule-based, not an LLM proposer; budget fixed at 40 runs; only 5
  held-out functions; setup costs (fit, memory warm-up, Self-HPO training) are outside the per-bug budget.
  Laptop adaptations preserve each paper's *idea*, not its scale or headline numbers.

## Code map
```
rsi_debug/corpus.py     programs, mutation operators, CI runner, bug benchmark
rsi_debug/agent.py      Harness, Policy, Memory, features, SO-RSI diagnostics, the episode loop
rsi_debug/scorer.py     Budget RSI: labelled patches + ridge linear readout
rsi_debug/self_hpo.py   Self-HPO: REINFORCE P-step, failure-driven H-step
rsi_debug/evaluate.py   seeds, summaries, per-bug vectors
```

## Video outline (what each video covers)
Each video: (1) state the paper's idea in one sentence, (2) open the code and walk the 2-3 key
functions, (3) run the script on screen, (4) read the output and state one limitation.

| Video | Show / run | Key code to walk through | Say out loud |
|---|---|---|---|
| 0 Pipeline + baseline (~6 min) | `part0_...py` | `corpus.py`: `candidates`, `apply_edit`, `run_tests`; `agent.py`: `run_episode` | what a patch is, why hidden tests exist, naive ~14-29% |
| 1 Budget RSI (~8 min) | `part1_...py` | `scorer.py` (both functions), `featurize` | steering split = visible tests; readout cheap but not free; ranking edge small, compounds |
| 2 SO-RSI (~8 min) | `part2_...py` | `agent.py` SO-RSI block (anomaly masks, probe, `Memory`), `symptom()` | 3 anomalies; dedupe-only ablation = 0 gain; memory is what helps; the EQ mistake |
| 3 Self-HPO (~9 min) | `part3_...py --quick` (show saved full output too) | `self_hpo.py`: `h_step`, `propose_harnesses`, `p_step` | H-step vs P-step; H-only does most; synergy not reproduced |
| 4 Comparison + claim (~6 min) | `part4_...py` (or its saved output) | bootstrap, the claim block | three objects; what the CIs allow and forbid |

Each course lab ends with a quiz: attempt it and explain one new case in the video.

## Reproducibility notes
Seeds are fixed; results above are from the saved files in `outputs/`. Re-running reproduces them
(CI results are cached for speed; the logical cost accounting is unchanged).
