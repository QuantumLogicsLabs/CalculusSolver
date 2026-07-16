# Training Results

**Best Validation Loss:** 0.6246
**Total Epochs Run:** 5

## Per-Epoch Metrics

| Epoch | Train Loss | Val Loss | Val Seq | Val Rule | Val Verify | Checkpoint Saved |
|-------|-----------|----------|---------|----------|------------|-----------------|
| 1 | 0.6725 | 0.6524 | 0.0070 | 0.6421 | 0.0034 | Yes |
| 2 | 0.6403 | 0.6298 | 0.0036 | 0.6243 | 0.0020 | Yes |
| 3 | 0.6538 | 0.6246 | 0.0022 | 0.6211 | 0.0013 | Yes |
| 4 | 0.6380 | 0.6261 | 0.0016 | 0.6237 | 0.0008 | No |
| 5 | 0.6341 | 0.6328 | 0.0011 | 0.6310 | 0.0007 | No |

## Configuration

- **Learning Rate:** 0.0001
- **Batch Size:** 32
- **Hidden Dim:** 128
- **Max Steps/Epoch:** 100
- **Early Stopping:** patience=5, min_delta=0.001
- **Vocab Size:** 100
- **Num Rules:** 10
