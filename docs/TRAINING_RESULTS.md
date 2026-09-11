# Training Results

**Git Commit Hash:** `83e1a7266b605161bd202b58cb29a8dc55dc4be2`
**Best Validation Loss:** 0.0045
**Total Epochs Run:** 15

## Per-Epoch Metrics

| Epoch | Train Loss | Val Loss | Per-Token Acc | Val Seq Acc | Saved |
|-------|-----------|----------|---------------|-------------|-------|
| 1 | 0.2021 | 0.0127 | 0.9956 | 0.9418 | Yes |
| 2 | 0.0108 | 0.0089 | 0.9966 | 0.9515 | Yes |
| 3 | 0.0084 | 0.0069 | 0.9972 | 0.9581 | Yes |
| 4 | 0.0073 | 0.0063 | 0.9974 | 0.9587 | Yes |
| 5 | 0.0066 | 0.0057 | 0.9977 | 0.9623 | Yes |
| 6 | 0.0063 | 0.0053 | 0.9977 | 0.9625 | Yes |
| 7 | 0.0059 | 0.0053 | 0.9977 | 0.9617 | No |
| 8 | 0.0055 | 0.0049 | 0.9977 | 0.9626 | Yes |
| 9 | 0.0053 | 0.0047 | 0.9977 | 0.9612 | Yes |
| 10 | 0.0056 | 0.0046 | 0.9978 | 0.9636 | No |
| 11 | 0.0055 | 0.0046 | 0.9978 | 0.9635 | No |
| 12 | 0.0053 | 0.0054 | 0.9974 | 0.9599 | No |
| 13 | 0.0053 | 0.0046 | 0.9974 | 0.9563 | No |
| 14 | 0.0053 | 0.0045 | 0.9975 | 0.9588 | Yes |
| 15 | 0.0054 | 0.0103 | 0.9954 | 0.9435 | No |

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
