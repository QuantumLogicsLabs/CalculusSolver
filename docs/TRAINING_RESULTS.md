# Training Results

**Git Commit Hash:** `17e1e1ad1a3682d45768505fa30fc32266151e04`
**Best Validation Loss:** 0.0916
**Total Epochs Run:** 8

## Per-Epoch Metrics

| Epoch | Train Loss | Val Loss | Per-Token Acc | Val Seq Acc | Free-Run Seq Acc | Free-Run Token Acc | Avg Gen Len | Saved |
|-------|-----------|----------|---------------|-------------|------------------|--------------------|-------------|-------|
| 1 | 0.4837 | 0.1287 | 0.9530 | 0.1947 | 0.0000 | 0.4542 | 20.5 | Yes |
| 2 | 0.1401 | 0.1000 | 0.9611 | 0.2376 | 0.0000 | 0.4654 | 20.7 | Yes |
| 3 | 0.1257 | 0.0916 | 0.9630 | 0.2515 | 0.0667 | 0.5962 | 19.3 | Yes |
| 4 | 0.1220 | 0.0923 | 0.9635 | 0.2576 | 0.1333 | 0.5581 | 18.7 | No |
| 5 | 0.1221 | 0.0928 | 0.9642 | 0.2603 | 0.0667 | 0.4496 | 21.7 | No |
| 6 | 0.1241 | 0.0968 | 0.9622 | 0.2397 | 0.0667 | 0.4577 | 23.1 | No |
| 7 | 0.1183 | 0.0939 | 0.9631 | 0.2467 | 0.0667 | 0.4462 | 21.9 | No |
| 8 | 0.1154 | 0.0921 | 0.9637 | 0.2475 | 0.0667 | 0.4346 | 23.7 | No |

## Per-Category Validation Sequence Accuracy

| Epoch | Diff | Gradient | Integrate | Partial | Tangent_line |
|-------|--- | --- | --- | --- | ---|
| 1 | 0.1970 | 0.1801 | 0.0418 | 0.2333 | 0.3413 |
| 2 | 0.2415 | 0.2197 | 0.1566 | 0.2575 | 0.3889 |
| 3 | 0.2881 | 0.2176 | 0.1691 | 0.2707 | 0.4127 |
| 4 | 0.3051 | 0.2122 | 0.1858 | 0.2747 | 0.4524 |
| 5 | 0.3008 | 0.2069 | 0.2025 | 0.2793 | 0.4603 |
| 6 | 0.2903 | 0.1704 | 0.1649 | 0.2678 | 0.4603 |
| 7 | 0.2924 | 0.1961 | 0.1816 | 0.2638 | 0.4603 |
| 8 | 0.2839 | 0.1876 | 0.1795 | 0.2718 | 0.4762 |

## Configuration Snapshot

- **Learning Rate:** 0.0001
- **Warmup Steps:** 1000
- **Batch Size:** 32
- **Hidden Dim:** 256
- **Max Len:** 80
- **Max Steps/Epoch:** 3500
- **Early Stopping:** patience=6, min_delta=0.0001
- **Vocab Size:** 126
- **Gradient Clipping:** max_norm=1.0
- **Rule Prediction:** Multi-head prediction output (decoder_logits, rule_logits, verifier_logits)
