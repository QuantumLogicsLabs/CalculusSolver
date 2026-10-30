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

