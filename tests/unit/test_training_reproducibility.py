"""Seeding, run manifest, and exact-match metrics."""

import json
import random

import pytest
import torch

from training.metrics import (
    contrast_pair_accuracy,
    evaluate_exact_match,
    greedy_decode_batch,
    select_indices,
)
from training.reproducibility import (
    build_run_manifest,
    collect_environment,
    describe_files,
    set_global_seed,
    sha256_file,
    write_json,
)


def _draw():
    return random.random(), torch.rand(3).tolist(), torch.randint(0, 1000, (3,)).tolist()


def test_seed_reproduces_python_and_torch_streams():
    set_global_seed(123, deterministic=False)
    first = _draw()
    set_global_seed(123, deterministic=False)
    assert _draw() == first
    set_global_seed(124, deterministic=False)
    assert _draw() != first


def test_manifest_records_everything_needed_to_reproduce(tmp_path):
    vocab = tmp_path / "vocab.json"
    vocab.write_text(json.dumps({"_version": 9.9, "special_tokens": {"[PAD]": 0}}), encoding="utf-8")
    serializer = tmp_path / "ser.py"
    serializer.write_text("x = 1\n", encoding="utf-8")
    data = tmp_path / "train.jsonl"
    data.write_text('{"a": 1}\n{"a": 2}\n', encoding="utf-8")

    manifest = build_run_manifest(
        run_name="t",
        cfg={"epochs": 15},
        seed_info=set_global_seed(5, deterministic=False),
        dataset_files=[data, tmp_path / "missing.jsonl"],
        vocab_path=str(vocab),
        serializer_path=str(serializer),
        vocab_size=1,
        checkpoint_path="x/best.pt",
        optimizer_info={"name": "AdamW"},
        scheduler_info={"type": "cosine"},
        loss_info={"token_weighted": True},
    )
    assert manifest["seeds"]["seed"] == 5
    assert manifest["tokenizer"]["vocab_version"] == 9.9
    assert manifest["tokenizer"]["vocab_sha256"] == sha256_file(vocab)
    assert manifest["checkpoint"]["path"] == "x/best.pt"
    files = manifest["dataset"]
    assert files[data.as_posix()]["rows"] == 2 and files[data.as_posix()]["sha256"]
    assert files[(tmp_path / "missing.jsonl").as_posix()]["exists"] is False
    assert {"python", "platform", "packages", "git_commit", "device_name"} <= set(manifest["environment"])
    for key in ("optimizer", "scheduler", "loss", "config"):
        assert key in manifest

    out = tmp_path / "m" / "run_manifest.json"
    write_json(out, manifest)
    assert json.loads(out.read_text(encoding="utf-8"))["run_name"] == "t"


def test_environment_reports_device_and_versions():
    env = collect_environment()
    assert env["packages"]["torch"] == torch.__version__
    assert env["device_name"]


def test_describe_files_hashes_change_with_content(tmp_path):
    f = tmp_path / "a.jsonl"
    f.write_text("1\n", encoding="utf-8")
    h1 = describe_files([f])[f.as_posix()]["sha256"]
    f.write_text("2\n", encoding="utf-8")
    assert describe_files([f])[f.as_posix()]["sha256"] != h1


# -- exact match --------------------------------------------------------------

PAD, BOS, EOS = 0, 1, 2


class _ScriptedModel(torch.nn.Module):
    """Emits a fixed token script per batch row, so decoding is predictable."""

    def __init__(self, scripts, vocab=8):
        super().__init__()
        self.scripts = scripts
        self.vocab = vocab
        self.anchor = torch.nn.Parameter(torch.zeros(1))

    def forward(self, src, tgt_in, true_rule_ids=None):
        step = tgt_in.size(1) - 1
        logits = torch.zeros(src.size(0), tgt_in.size(1), self.vocab)
        for row, script in enumerate(self.scripts):
            tok = script[step] if step < len(script) else EOS
            logits[row, -1, tok] = 10.0
        return logits, None, None


class _ToyData:
    def __init__(self, ops, targets, groups=None):
        self.ops = ops
        self.targets = targets
        self.contrast_groups = groups or [None] * len(ops)

    def __len__(self):
        return len(self.ops)

    def __getitem__(self, i):
        tgt = [BOS] + self.targets[i] + [EOS]
        tgt += [PAD] * (8 - len(tgt))
        return {
            "src_seq": torch.tensor([3, 4, 5]),
            "tgt_out_seq": torch.tensor(tgt),
            "op": self.ops[i],
        }


def test_exact_match_scores_whole_sequences_by_operation():
    data = _ToyData(
        ops=["diff", "diff", "partial"],
        targets=[[3, 4], [3, 4], [5, 6, 7]],
    )
    model = _ScriptedModel(scripts=[[3, 4], [3, 5], [5, 6, 7]])
    res = evaluate_exact_match(
        model, data, bos_id=BOS, eos_id=EOS, pad_id=PAD, max_gen_len=6, batch_size=8
    )
    assert res["overall"] == {"correct": 2, "total": 3, "accuracy": pytest.approx(2 / 3)}
    assert res["by_op"]["diff"]["correct"] == 1 and res["by_op"]["diff"]["total"] == 2
    assert res["by_op"]["partial"]["accuracy"] == 1.0
    assert res["correct_flags"] == {0: True, 1: False, 2: True}
    # token accuracy is higher than exact match: the wrong diff row still got token 3 right
    assert res["token_accuracy"] > res["overall"]["accuracy"]


def test_over_long_generation_is_not_counted_correct():
    data = _ToyData(ops=["diff"], targets=[[3, 4]])
    model = _ScriptedModel(scripts=[[3, 4, 3, 4, 3, 4]])  # never reaches EOS early
    res = evaluate_exact_match(model, data, bos_id=BOS, eos_id=EOS, pad_id=PAD, max_gen_len=4)
    assert res["overall"]["correct"] == 0


def test_contrast_pair_accuracy_needs_both_members():
    data = _ToyData(
        ops=["partial"] * 4,
        targets=[[3], [4], [5], [6]],
        groups=[0, 0, 1, 1],
    )
    model = _ScriptedModel(scripts=[[3], [4], [5], [7]])  # pair 1 second member wrong
    res = evaluate_exact_match(model, data, bos_id=BOS, eos_id=EOS, pad_id=PAD, max_gen_len=4)
    stats = contrast_pair_accuracy(res, data.contrast_groups)
    assert stats["pairs"] == 2 and stats["pairs_both_correct"] == 1
    assert stats["pair_accuracy"] == 0.5
    assert stats["row_accuracy"] == 0.75


def test_select_indices_is_stratified_and_deterministic():
    ops = ["a"] * 50 + ["b"] * 50 + ["c"] * 5
    chosen = select_indices(ops, 30)
    assert chosen == select_indices(ops, 30)
    assert {ops[i] for i in chosen} == {"a", "b", "c"}
    assert select_indices(ops, None) == list(range(len(ops)))
    assert select_indices(ops, 10_000) == list(range(len(ops)))


def test_greedy_decode_stops_when_all_rows_finish():
    model = _ScriptedModel(scripts=[[3], [4]])
    out = greedy_decode_batch(model, torch.zeros(2, 3, dtype=torch.long), BOS, EOS, PAD, 10)
    assert out.size(1) == 2  # one token then EOS, then the loop stops
