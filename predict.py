import sys
import json
import torch
from pathlib import Path

sys.path.insert(0, str(Path(__file__).parent.resolve()))

from solver_model import CalculusSolverModel
from tokenizer.slang_serializer import serialize_slang_math


def flatten_vocab(raw_vocab):
    """
    Identical to train.py's flatten_vocab — must stay identical, or token IDs
    at inference time will drift from the IDs the checkpoint was trained on.
    """
    flat = {}
    for key, value in raw_vocab.items():
        if key.startswith("_"):
            continue
        if isinstance(value, dict):
            flat.update(value)
    return flat


with open("tokenizer/vocab.json", "r", encoding="utf-8") as f:
    _raw_vocab = json.load(f)

vocab_mapping = flatten_vocab(_raw_vocab)
REAL_VOCAB_SIZE = max(vocab_mapping.values()) + 1

# Same rule label derivation as train.py, so RuleHead's output layer is the
# same shape the checkpoint was trained with.
_rule_items = sorted(_raw_vocab.get("rule_tokens", {}).items(), key=lambda kv: kv[1])
RULE_LABELS = [name.split("RULE:", 1)[1] for name, _ in _rule_items]

with open("config.json", "r") as cfg_file:
    config = json.load(cfg_file)

MAX_LEN = config.get("max_len", 32)


def encode(tokens):
    pad_idx = vocab_mapping["[PAD]"]
    ids = []
    for t in tokens:
        if t not in vocab_mapping:
            raise KeyError(f"Token '{t}' missing from vocab.json!")
        ids.append(vocab_mapping[t])
    pad_len = MAX_LEN - len(ids)
    if pad_len > 0:
        ids += [pad_idx] * pad_len
    return ids[:MAX_LEN]


def evaluate_cli_input():
    if len(sys.argv) < 2:
        print("💡 Usage: python predict.py '{\"op\": \"diff\", \"var\": \"x\", \"expr\": {\"coeff\": 3, \"var\": {\"x\": 2}}}'")
        return

    try:
        user_envelope = json.loads(sys.argv[1])
    except Exception:
        print("❌ Error: Command line argument must be valid JSON.")
        return

    try:
        src_tokens = serialize_slang_math(user_envelope)
    except Exception as e:
        print(f"❌ Error: Input is not a valid SLaNg envelope: {e}")
        return

    try:
        src_ids = encode(src_tokens)
    except KeyError as e:
        print(f"❌ Error: {e}")
        return

    src_tensor = torch.tensor([src_ids], dtype=torch.long)

    bos_id = vocab_mapping["[BOS]"]
    tgt_tensor = torch.full((1, MAX_LEN), vocab_mapping["[PAD]"], dtype=torch.long)
    tgt_tensor[0, 0] = bos_id

    model = CalculusSolverModel(
        vocab_size=REAL_VOCAB_SIZE,
        num_rules=len(RULE_LABELS),
        hidden_dim=config["hidden_dim"],
    )

    checkpoint_path = "checkpoints/checkpoint_epoch_1.pt"
    state_dict = torch.load(checkpoint_path, map_location="cpu")
    model.load_state_dict(state_dict)  # let mismatches raise loudly, never swallow them
    model.eval()

    with torch.no_grad():
        decoder_logits, rule_logits, verifier_logits = model(
            src_tensor,
            tgt_tensor,
        )
        pred_rule_idx = torch.argmax(rule_logits, dim=-1).item()
        pred_rule_name = RULE_LABELS[pred_rule_idx] if pred_rule_idx < len(RULE_LABELS) else str(pred_rule_idx)
        pred_token_ids = torch.argmax(decoder_logits, dim=-1)[0].tolist()
        confidence = torch.sigmoid(verifier_logits).mean().item()

    id_to_token = {v: k for k, v in vocab_mapping.items()}
    pred_tokens = [id_to_token.get(i, f"<id:{i}>") for i in pred_token_ids]

    print(f"\n🎯 Predicted rule: {pred_rule_name}")
    print(f"   Verification confidence: {confidence * 100:.2f}%")
    print(f"   Predicted output tokens: {pred_tokens}")


if __name__ == "__main__":
    evaluate_cli_input()