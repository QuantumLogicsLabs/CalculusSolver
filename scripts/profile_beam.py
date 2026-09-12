"""Profile beam-search cost: is_valid_prefix() call volume and where the
wall clock actually goes, across beam widths, with and without the
repetition guards.

Answers DEV 2 task 3. Re-run this whenever the grammar, the guards, or the
beam bookkeeping changes.

Method
------
Forward-pass COUNT is exact (every model call is counted). Forward-pass
WALL TIME is projected from a separately measured per-call cost on the real
architecture -- timing 300 problems x N beams x 50ms in-loop would take
tens of minutes and measure nothing extra, since the cost per call does not
depend on weight values. Mask time is measured in-loop and is real.

Two model behaviours bracket reality, because decode LENGTH is
model-dependent and there is no checkpoint in this repo:

  oracle    - certain about every gold token; terminates correctly.
              The best case, and what a trained model should approach.
  untrained - uniform logits; wanders until max_len.
              The worst case, and what the current checkpoint does.

Usage:
    python scripts/profile_beam.py [--problems N] [--beams 1,2,4,5,8]
"""

import argparse
import glob
import json
import random
import sys
import time
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

import torch  # noqa: E402

import inference.grammar as grammar  # noqa: E402
from inference.beam_search import beam_search  # noqa: E402
from inference.grammar import EOS_TOKEN, load_vocab  # noqa: E402
from tokenizer.slang_serializer import serialize_slang_math  # noqa: E402

torch.set_grad_enabled(False)
torch.set_num_threads(4)

VOCAB = load_vocab(str(ROOT / "tokenizer" / "vocab.json"))
VOCAB_SIZE = max(VOCAB["id_to_token"]) + 1
TID = VOCAB["token_to_id"]

STATS = {"calls": 0, "tokens": 0, "mask_s": 0.0}
_real_is_valid_prefix = grammar.is_valid_prefix


def _counting_is_valid_prefix(tokens):
    STATS["calls"] += 1
    STATS["tokens"] += len(tokens)
    return _real_is_valid_prefix(tokens)


class CountingPool(grammar.NodeValidityPool):
    """Times mask() in-loop without perturbing what it computes."""

    def mask(self, tokens, candidate_tokens):
        t0 = time.perf_counter()
        out = super().mask(tokens, candidate_tokens)
        STATS["mask_s"] += time.perf_counter() - t0
        return out


class OracleModel:
    def __init__(self, gold_ids):
        self.gold_ids = list(gold_ids)
        self.calls = 0

    def __call__(self, src_tokens, tgt):
        self.calls += 1
        step = tgt.shape[1] - 1
        logits = torch.full((1, tgt.shape[1], VOCAB_SIZE), -8.0)
        if step < len(self.gold_ids):
            logits[0, -1, self.gold_ids[step]] = 20.0
        return (logits, None, None)


class UntrainedModel:
    def __init__(self, seed=0):
        self.rng = torch.Generator().manual_seed(seed)
        self.calls = 0

    def __call__(self, src_tokens, tgt):
        self.calls += 1
        logits = torch.randn(
            (1, tgt.shape[1], VOCAB_SIZE), generator=self.rng
        )
        return (logits, None, None)


def measure_forward_cost(samples=12):
    """Per-forward cost on the real architecture (random weights: compute is
    identical regardless of weight values)."""
    from model.transformer import CalculusSolverModel

    config = json.loads((ROOT / "config.json").read_text())
    model = CalculusSolverModel(
        vocab_size=VOCAB_SIZE, num_rules=13, hidden_dim=config["hidden_dim"]
    ).eval()
    src = torch.zeros(1, 48, dtype=torch.long)
    tgt = torch.zeros(1, 20, dtype=torch.long)
    model(src, tgt)
    t0 = time.perf_counter()
    for _ in range(samples):
        model(src, tgt)
    return (time.perf_counter() - t0) / samples


def load_problems(limit=None):
    problems = []
    for path in sorted(glob.glob(str(ROOT / "eval" / "benchmarks" / "*.json"))):
        op = Path(path).stem.replace("benchmark_", "")
        for row in json.load(open(path, encoding="utf-8")):
            target = row.get("target")
            if target is None:
                continue
            try:
                gold = serialize_slang_math(target)
            except Exception:
                continue
            problems.append((op, gold))
    if limit and limit < len(problems):
        random.Random(0).shuffle(problems)
        problems = problems[:limit]
    return problems


def profile(problems, beam_size, max_len, guards, behaviour):
    STATS.update(calls=0, tokens=0, mask_s=0.0)
    pool = CountingPool()
    forwards = 0
    t0 = time.perf_counter()
    for _, gold in problems:
        if behaviour == "oracle":
            model = OracleModel([TID[t] for t in gold + [EOS_TOKEN] if t in TID])
        else:
            model = UntrainedModel()
        kwargs = {} if guards else {
            "max_token_run": 0, "no_repeat_ngram_size": 0, "repetition_penalty": 1.0,
        }
        beam_search(
            model=model, src_tokens=torch.zeros(1, 48, dtype=torch.long),
            vocab_map=VOCAB, beam_size=beam_size, max_len=max_len,
            node_pool=pool, **kwargs,
        )
        forwards += model.calls
    wall = time.perf_counter() - t0
    n = len(problems)
    return {
        "ivp_calls": STATS["calls"] / n,
        "ivp_tokens": STATS["tokens"] / n,
        "mask_s": STATS["mask_s"] / n,
        "forwards": forwards / n,
        "harness_wall": wall / n,
    }


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--problems", type=int, default=60)
    ap.add_argument("--beams", default="1,2,4,5,8")
    ap.add_argument("--max-len", type=int, default=32)
    args = ap.parse_args()

    grammar.is_valid_prefix = _counting_is_valid_prefix
    problems = load_problems(args.problems)
    beams = [int(b) for b in args.beams.split(",")]

    fwd_cost = measure_forward_cost()
    print(f"problems profiled      : {len(problems)}")
    print(f"max_len                : {args.max_len}")
    print(f"measured forward cost  : {fwd_cost * 1000:.2f} ms/call "
          f"(CalculusSolverModel, hidden_dim=256, 4 threads)\n")

    for behaviour in ("oracle", "untrained"):
        print(f"### {behaviour} model")
        print("| beam | guards | is_valid_prefix calls | tokens parsed | "
              "mask ms | forwards | proj. fwd s | proj. total s | mask share |")
        print("|---|---|---|---|---|---|---|---|---|")
        for beam_size in beams:
            for guards in (True, False):
                r = profile(problems, beam_size, args.max_len, guards, behaviour)
                fwd_s = r["forwards"] * fwd_cost
                total = fwd_s + r["mask_s"]
                print(
                    f"| {beam_size} | {'on' if guards else 'off'} | "
                    f"{r['ivp_calls']:,.0f} | {r['ivp_tokens']:,.0f} | "
                    f"{r['mask_s'] * 1000:,.0f} | {r['forwards']:.0f} | "
                    f"{fwd_s:.2f} | {total:.2f} | "
                    f"{r['mask_s'] / total * 100:.2f}% |"
                )
        print()


if __name__ == "__main__":
    main()
