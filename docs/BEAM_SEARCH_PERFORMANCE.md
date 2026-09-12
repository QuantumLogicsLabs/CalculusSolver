# Beam Search Performance — validity-mask cost and the eval runtime bound

Answers DEV 2 task 3. Reproduce with:

```bash
python scripts/profile_beam.py --problems 8 --beams 1,2,5 --max-len 256
```

## Method

Forward-pass **count** is exact. Forward-pass **wall time** is projected from a
separately measured per-call cost on the real architecture
(`CalculusSolverModel`, `hidden_dim=256`, 4 threads, `src_len=48`,
**~24 ms/call**); timing every call in-loop would add tens of minutes and
measure nothing extra, since cost per call does not depend on weight values.
Mask time is measured in-loop and is real.

There is no checkpoint in this repo, and decode **length** is model-dependent,
so two behaviours bracket reality:

| behaviour | meaning |
|---|---|
| **oracle** | certain about every gold token; terminates correctly. What a trained model should approach. |
| **untrained** | uniform random logits; wanders. The pessimistic bound. |

## 1. Validity-mask cost at beam_size=5

All figures at `max_len=256`, which is what `run_eval.py` actually uses
(see §3).

| behaviour | beam | guards | is_valid_prefix calls | tokens parsed | mask ms | forwards | proj. total s | **mask share** |
|---|---|---|---|---|---|---|---|---|
| oracle | 1 | on | 2,806 | 36,410 | 27 | 23 | 0.60 | 4.4% |
| oracle | 2 | on | 5,487 | 72,695 | 44 | 44 | 1.17 | 3.8% |
| oracle | 5 | on | 13,532 | 181,552 | 120 | 109 | 2.89 | 4.2% |
| untrained | 1 | on | 6,076 | 151,900 | 78 | 49 | 1.32 | 5.9% |
| untrained | 2 | on | 15,252 | 484,220 | 205 | 123 | 3.32 | 6.2% |
| untrained | 5 | on | 27,404 | 641,204 | 337 | 221 | 5.94 | 5.7% |

**The mask is not the bottleneck.** It is 4–6% of runtime; the un-batched
forward passes are the rest. Beam width scales both terms roughly linearly, so
`beam_size=2 -> 5` costs about 2.5x across the board — the mask is not what
makes width expensive.

## 2. Effect of the repetition guards on mask cost

The guards never increase `is_valid_prefix` volume. They are either neutral or
a small saving:

| behaviour | beam | guards on | guards off |
|---|---|---|---|
| oracle | 5 | 13,532 | 13,532 (identical) |
| untrained | 1 | 6,076 | 6,448 |
| untrained | 5 | 27,404 | 29,884 |

Their own cost is **0.0084 ms/step** — 0.02% of a step, against ~24 ms for a
forward pass.

> **Correction.** An earlier revision of this document claimed call volume was
> *identical* with guards on and off, and described that as structural. That was
> over-generalised from two configurations. Volume is driven by
> `beams x steps x |V|`, and the guards do change how many steps a degenerate
> beam survives — so they can reduce it. The claim to rely on is the weaker,
> true one: **the guards never make mask cost worse.**

## 3. The `eval/run_eval.py` bound

`run_eval.py` constructs `CalculusSolverInference(model_path=..., beam_size=5)`
with no `max_len`, so it inherits **256** from `inference/solve.py`. Over the
300 benchmark problems:

| model behaviour | per problem | **300 problems** |
|---|---|---|
| correct (terminates) | 2.89 s | **~15 minutes** |
| worst case (never terminates) | 5.94 s | **~30 minutes** |

**Documented bound: `run_eval.py` at `beam_size=5` should complete in ~15
minutes, and must not exceed ~30 minutes.** Materially longer than that means
something regressed in the grammar bounds of §4 — it is not the mask.

This bound is projected, not measured end-to-end: `run_eval.py` exits
immediately because `checkpoints/final/best.pt` is absent. Re-run
`scripts/profile_beam.py` once a checkpoint exists to close the gap.

## 4. Structural bounds are what made the worst case tractable

Before the grammar was bounded, the untrained worst case at `beam_size=5`,
`max_len=256` was **34.58 s/problem** — 158,224 `is_valid_prefix` calls and
**20.4 million tokens parsed**, with the mask at 14.7%. `is_valid_prefix`
re-parses the whole prefix per candidate, so cost is
`O(beams x steps x |V| x L)` — quadratic in sequence length. A decoder that
never terminates drags every beam to `L=256` and pays that quadratic.

Capping the grammar's previously unbounded dimensions removed it:

| bound | max in 155,000 real targets | cap | margin |
|---|---|---|---|
| `MAX_OPVARS_PER_OP` | 1 | 4 | 4x |
| `MAX_NESTING_DEPTH` | 3 | 8 | 2.7x |
| `MAX_SIBLINGS` | 2 | 8 | 4x |

Result: **34.58 s -> 5.94 s per problem** (5.8x), tokens parsed 20.4M -> 641k
(32x). Grammar acceptance of real targets is unchanged at **100.00%** of all
155,000 — the caps reject nothing real.

## 5. Known limitation

The bounds do **not** guarantee that a degenerate decoder terminates within
`max_len`. A tree of depth 8 with 8 siblings per node can hold far more than
256 tokens, so a pathological model can still fill the budget with a legal but
incomplete prefix and be truncated. Measured: a model biased toward nested
`NODE:FRAC` still reaches `max_len` with `status="partial"`.

`beam_search` mitigates but cannot eliminate this: when no beam terminates it
returns the longest **closed** prefix (`longest_complete_prefix`), so the caller
gets something parseable. That only helps when the outermost node closed — a
sequence that opens a fraction and never closes it has no complete prefix, and
is returned as-is.

Guaranteeing termination would require a budget-aware constraint: once the
remaining budget equals the tokens needed to close all open structures, mask
everything except closing tokens. That is implementable but changes what the
model is allowed to emit, so it is flagged rather than assumed.

## 6. Recommendation

**Decouple the decoder step budget from the encoder padding width.**
`inference/solve.py` uses one `max_len=256` for both source padding
(`solve.py:137-139`) and the generation budget (`solve.py:155`). They are
unrelated quantities.

The longest serialized target across all 155,000 dataset targets is **38
tokens**. A decoder budget of **48** (already the value in `config.json`) covers
100% of real targets with 9 tokens spare. This caps the worst case directly
rather than relying on the model to stop.

Secondary: batching the beams into one forward call per step was measured at
only **~1.5x** (373ms -> 254ms for 5 beams), not the 5x it looks like — the
model is CPU-bound and does not parallelise well across batch at 4 threads.
Worth doing, but it is not the lever.
