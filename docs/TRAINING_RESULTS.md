# Training Results

**Git Commit Hash:** `b44635d12412ecbc8e34e1f0845a2bf3050b264f`
**Best Validation Loss:** 0.0053
**Total Epochs Run:** 6

## Per-Epoch Metrics

| Epoch | Train Loss | Val Loss | Per-Token Acc | Val Seq Acc | Saved |
|-------|-----------|----------|---------------|-------------|-------|
| 1 | 0.2052 | 0.0122 | 0.9952 | 0.9388 | Yes |
| 2 | 0.0109 | 0.0077 | 0.9966 | 0.9517 | Yes |
| 3 | 0.0080 | 0.0088 | 0.9970 | 0.9519 | No |
| 4 | 0.0072 | 0.0064 | 0.9974 | 0.9597 | Yes |
| 5 | 0.0068 | 0.0053 | 0.9977 | 0.9627 | Yes |
| 6 | 0.0061 | 0.0051 | 0.9976 | 0.9625 | No |

## Configuration Snapshot

- **Learning Rate:** 0.0001
- **Warmup Steps:** 1000
- **Batch Size:** 32
- **Hidden Dim:** 256
- **Max Len:** 48
- **Max Steps/Epoch:** 3500
- **Early Stopping:** patience=15, min_delta=0.0002
- **Vocab Size:** 124
- **Gradient Clipping:** max_norm=1.0
- **Rule Prediction:** Multi-head prediction output (decoder_logits, rule_logits, verifier_logits)
