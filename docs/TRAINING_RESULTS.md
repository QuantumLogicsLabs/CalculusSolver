# Training Results

**Git Commit Hash:** `test_commit`
**Best Validation Loss:** 0.0500
**Total Epochs Run:** 1

## Per-Epoch Metrics

| Epoch | Train Loss | Val Loss | Per-Token Acc | Val Seq Acc | Free-Run Seq Acc | Free-Run Token Acc | Avg Gen Len | Saved |
|-------|-----------|----------|---------------|-------------|------------------|--------------------|-------------|-------|
| 1 | 0.1000 | 0.0500 | 0.9000 | 0.8000 | 0.0000 | 0.0000 | 0.0 | Yes |

## Configuration Snapshot

- **Learning Rate:** 0.0001
- **Warmup Steps:** 1000
- **Batch Size:** 32
- **Hidden Dim:** 256
- **Max Len:** 48
- **Max Steps/Epoch:** 3500
- **Early Stopping:** False
- **Vocab Size:** 126
- **Gradient Clipping:** max_norm=1.0
- **Rule Prediction:** Multi-head prediction output (decoder_logits, rule_logits, verifier_logits)
