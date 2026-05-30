import json
import random
import subprocess
from pathlib import Path
from typing import Any, Dict, Iterable, List, Optional, Sequence, Tuple

import torch
from torch.utils.data import Dataset


def load_yaml(path: Path) -> Dict[str, Any]:
    try:
        import yaml
    except ImportError as exc:
        raise ImportError(
            "PyYAML is required to read YAML config files. Install with `pip install pyyaml`."
        ) from exc

    with path.open("r", encoding="utf-8") as handle:
        return yaml.safe_load(handle)


def flatten_vocab(vocab: Dict[str, Any]) -> Tuple[Dict[str, int], Dict[int, str]]:
    token_to_id = {}
    id_to_token = {}

    def _walk(node: Any, current_key: str = ""):
        if isinstance(node, dict):
            for key, value in node.items():
                _walk(value, key)
        elif isinstance(node, int):
            token = current_key
            token_to_id[token] = node
            id_to_token[node] = token

    _walk(vocab)
    return token_to_id, id_to_token


def load_vocab(vocab_path: Path) -> Tuple[Dict[str, int], Dict[int, str], List[str]]:
    with vocab_path.open("r", encoding="utf-8") as handle:
        raw_vocab = json.load(handle)

    token_to_id, id_to_token = flatten_vocab(raw_vocab)
    rule_labels = []
    for token in raw_vocab.get("rule_tokens", {}).keys():
        if token.startswith("RULE:"):
            rule_labels.append(token.split("RULE:", 1)[-1])
        else:
            rule_labels.append(token)
    return token_to_id, id_to_token, rule_labels


def resolve_jsonl_path(path: Path) -> List[Path]:
    if path.is_dir():
        return sorted(path.glob("*.jsonl"))

    if path.is_file():
        return [path]

    jsonl_path = path.with_suffix(".jsonl")
    if jsonl_path.is_file():
        return [jsonl_path]

    raise FileNotFoundError(f"Could not resolve JSONL path: {path}")


def load_jsonl(path: Path) -> List[Dict[str, Any]]:
    examples: List[Dict[str, Any]] = []
    for file_path in resolve_jsonl_path(path):
        with file_path.open("r", encoding="utf-8") as handle:
            for line in handle:
                line = line.strip()
                if not line:
                    continue
                examples.append(json.loads(line))
    return examples


def serialize_slang_object(obj: Any, script_path: Path) -> List[str]:
    payload = json.dumps(obj, ensure_ascii=False)
    process = subprocess.run(
        ["node", str(script_path)],
        input=payload.encode("utf-8"),
        stdout=subprocess.PIPE,
        stderr=subprocess.PIPE,
        check=False,
    )
    if process.returncode != 0:
        raise RuntimeError(
            f"Node serializer failed: {process.stderr.decode('utf-8', errors='replace')}"
        )
    output = process.stdout.decode("utf-8", errors="replace").strip()
    if not output:
        raise RuntimeError("Node serializer returned empty output.")
    result = json.loads(output)
    if isinstance(result, dict) and "tokens" in result:
        return result["tokens"]
    return result


def is_operator_token(token: str) -> bool:
    return token.startswith("OP:") or token in {"NODE:FRAC", "NODE:TERM"}


def mask_token_ids(
    token_ids: List[int],
    token_strings: List[str],
    mask_id: int,
    mask_ratio: float = 0.2,
    special_ids: Optional[Sequence[int]] = None,
) -> List[int]:
    special_ids = set(special_ids or [])
    candidate_positions = [
        i
        for i, token in enumerate(token_strings)
        if is_operator_token(token) and token_ids[i] not in special_ids
    ]
    if not candidate_positions:
        candidate_positions = [
            i for i, token_id in enumerate(token_ids) if token_id not in special_ids
        ]
    num_to_mask = max(1, int(len(candidate_positions) * mask_ratio))
    positions = random.sample(
        candidate_positions, min(num_to_mask, len(candidate_positions))
    )
    masked = list(token_ids)
    for index in positions:
        masked[index] = mask_id
    return masked


def pad_sequences(sequences: List[List[int]], pad_id: int) -> torch.LongTensor:
    max_len = max(len(seq) for seq in sequences)
    batch = torch.full((len(sequences), max_len), pad_id, dtype=torch.long)
    for i, seq in enumerate(sequences):
        batch[i, : len(seq)] = torch.tensor(seq, dtype=torch.long)
    return batch


def pad_positions(positions: List[torch.Tensor]) -> torch.Tensor:
    max_len = max(pos.size(0) for pos in positions)
    batch = torch.zeros((len(positions), max_len, 3), dtype=torch.float32)
    for i, pos in enumerate(positions):
        batch[i, : pos.size(0), :] = pos
    return batch


class SlangJsonlDataset(Dataset):
    def __init__(
        self,
        data_path: Path,
        vocab_path: Path,
        serializer_path: Path,
        mode: str = "pretrain",
        max_len: int = 256,
        mask_ratio: float = 0.2,
    ):
        self.examples = load_jsonl(data_path)
        self.token_to_id, self.id_to_token, self.rule_labels = load_vocab(vocab_path)
        self.rule_to_id = {label: idx for idx, label in enumerate(self.rule_labels)}
        self.serializer_path = serializer_path
        self.mode = mode
        self.max_len = max_len
        self.mask_ratio = mask_ratio
        self.pad_id = self.token_to_id.get("[PAD]", 0)
        self.bos_id = self.token_to_id.get("[BOS]", 1)
        self.eos_id = self.token_to_id.get("[EOS]", 2)
        self.mask_id = self.token_to_id.get("[MASK]", 3)
        self._cache: List[Dict[str, Any]] = [{} for _ in self.examples]

    def __len__(self) -> int:
        return len(self.examples)

    def _tokenize(self, obj: Any) -> Tuple[List[int], List[str]]:
        token_strings = serialize_slang_object(obj, self.serializer_path)
        token_ids = [
            self.token_to_id.get(tok, self.token_to_id.get("[UNK]", 0))
            for tok in token_strings
        ]
        return token_ids, token_strings

    def _collect_positions(self, token_ids: List[int]) -> torch.Tensor:
        return torch.zeros((len(token_ids), 3), dtype=torch.float32)

    def __getitem__(self, idx: int) -> Dict[str, Any]:
        cache = self._cache[idx]
        if not cache:
            example = self.examples[idx]
            input_ids, input_tokens = self._tokenize(example["input"])
            input_ids = input_ids[: self.max_len]
            input_tokens = input_tokens[: self.max_len]
            cache["input_ids"] = input_ids
            cache["input_tokens"] = input_tokens
            cache["input_positions"] = self._collect_positions(input_ids)

            if self.mode == "finetune":
                output_ids, output_tokens = self._tokenize(example["output"]["expr"] if isinstance(example["output"], dict) and "expr" in example["output"] else example["output"])  # type: ignore
                output_ids = output_ids[: self.max_len]
                output_tokens = output_tokens[: self.max_len]
                cache["output_ids"] = output_ids
                cache["output_tokens"] = output_tokens
                rule_label = example.get("output", {}).get("rule", "undefined")
                cache["rule_id"] = self.rule_to_id.get(
                    rule_label, self.rule_to_id.get("undefined", 0)
                )

        return cache

    def collate(self, batch: Sequence[Dict[str, Any]]) -> Dict[str, torch.Tensor]:
        if self.mode == "pretrain":
            input_ids = [item["input_ids"] for item in batch]
            input_tokens = [item["input_tokens"] for item in batch]
            masked_ids = [
                mask_token_ids(
                    ids,
                    tokens,
                    self.mask_id,
                    self.mask_ratio,
                    special_ids=[self.pad_id, self.bos_id, self.eos_id],
                )
                for ids, tokens in zip(input_ids, input_tokens)
            ]
            padded_src = pad_sequences(masked_ids, self.pad_id)
            padded_tgt = pad_sequences(input_ids, self.pad_id)
            src_positions = pad_positions(
                [torch.zeros((len(ids), 3), dtype=torch.float32) for ids in input_ids]
            )
            return {
                "src_tokens": padded_src,
                "src_positions": src_positions,
                "tgt_input_tokens": self._shift_right(padded_tgt),
                "tgt_output_tokens": padded_tgt,
                "src_padding_mask": padded_src == self.pad_id,
                "tgt_padding_mask": padded_tgt == self.pad_id,
            }

        input_ids = [item["input_ids"] for item in batch]
        output_ids = [item["output_ids"] for item in batch]
        rule_ids = [item["rule_id"] for item in batch]
        padded_src = pad_sequences(input_ids, self.pad_id)
        padded_tgt = pad_sequences(output_ids, self.pad_id)
        src_positions = pad_positions(
            [torch.zeros((len(ids), 3), dtype=torch.float32) for ids in input_ids]
        )
        return {
            "src_tokens": padded_src,
            "src_positions": src_positions,
            "tgt_input_tokens": self._shift_right(padded_tgt),
            "tgt_output_tokens": padded_tgt,
            "rule_ids": torch.tensor(rule_ids, dtype=torch.long),
            "src_padding_mask": padded_src == self.pad_id,
            "tgt_padding_mask": padded_tgt == self.pad_id,
        }

    def _shift_right(self, tokens: torch.LongTensor) -> torch.LongTensor:
        shifted = tokens.new_full(tokens.shape, self.pad_id)
        shifted[:, 1:] = tokens[:, :-1].clone()
        shifted[:, 0] = self.bos_id
        return shifted
