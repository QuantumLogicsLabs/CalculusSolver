# Training Results

**Best Validation Loss:** 0.0317
**Total Epochs Run:** 10

## Per-Epoch Metrics

| Epoch | Train Loss | Val Loss | Val Seq Accuracy | Checkpoint Saved |
|-------|-----------|----------|-------------------|-----------------|
| 1 | 0.0981 | 0.0322 | 0.7711 | Yes |
| 2 | 0.0330 | 0.0317 | 0.7730 | Yes |
| 3 | 0.0323 | 0.0316 | 0.7730 | No |
| 4 | 0.0324 | 0.0318 | 0.7705 | No |
| 5 | 0.0324 | 0.0317 | 0.7730 | No |
| 6 | 0.0317 | 0.0316 | 0.7730 | No |
| 7 | 0.0323 | 0.0316 | 0.7730 | No |
| 8 | 0.0318 | 0.0332 | 0.7676 | No |
| 9 | 0.0320 | 0.0315 | 0.7730 | No |
| 10 | 0.0318 | 0.0314 | 0.7730 | No |

## Configuration

- **Architecture:** SimpleCalculusModel (standard nn.Transformer encoder-decoder)
- **Learning Rate:** 0.0001
- **Batch Size:** 32
- **Hidden Dim:** 256
- **Max Steps/Epoch:** 3500
- **Early Stopping:** patience=8, min_delta=0.0005
- **Vocab Size:** 106
- **Gradient Clipping:** max_norm=1.0
- **Rule prediction:** folded into output sequence as leading RULE:xxx token (see docs/KNOWN_ISSUES.md)
