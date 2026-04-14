# Diagnostic Tool Results — TODO 2 & TODO 3

**Date**: 2026-03-02
**Experiments compared**: `baseline_v2` (alpha=0.0) vs `gap_v2` (alpha=0.8), both epoch 60, val split

---

## TODO 3: Confusion Matrix Comparison (`compare_experiments.py`)

**Command**: `python compare_experiments.py --baseline_dir work_dir/hockey/baseline_v2 --textguided_dir work_dir/hockey/gap_v2 --split val`

### Per-Class Comparison (Epoch 60, Val)

| Class                    | F1 Baseline | F1 Text-Guided | Delta  | Verdict |
|--------------------------|-------------|----------------|--------|---------|
| GLID_FORW                | 79.4%       | 81.1%          | +1.7%  | SAME    |
| ACCEL_FORW               | 82.6%       | 83.2%          | +0.6%  | SAME    |
| GLID_BACK                | 42.3%       | 42.8%          | +0.5%  | SAME    |
| ACCEL_BACK               | 44.7%       | 38.0%          | -6.7%  | WORSE   |
| TRANS_FORW_TO_BACK       | 60.8%       | 64.3%          | +3.5%  | BETTER  |
| TRANS_BACK_TO_FORW       | 42.6%       | 33.6%          | -9.0%  | WORSE   |
| POST_WHISTLE_GLIDING     | 22.7%       | 27.8%          | +5.1%  | BETTER  |
| FACEOFF_BODY_POSITION    | 71.3%       | 75.4%          | +4.1%  | BETTER  |
| MAINTAIN_POSITION        | 27.3%       | 22.4%          | -4.9%  | WORSE   |
| PRONE                    | 16.7%       | 16.7%          | +0.0%  | SAME    |
| ON_A_KNEE                | 0.0%        | 0.0%           | +0.0%  | SAME    |
| **MACRO AVERAGE**        | **44.6%**   | **44.1%**      | **-0.5%** | |

**Overall Top-1 Accuracy**: Baseline=40.46% vs Text-Guided=40.53% (Delta=+0.08%)

### Key Takeaways
- Text guidance provides **no meaningful overall improvement** (+0.08% Top-1, -0.5% macro F1).
- Mixed per-class effects: 3 classes improve (TRANS_F2B, POST_WHISTLE, FACEOFF), 3 classes degrade (ACCEL_BACK, TRANS_B2F, MAINTAIN_POS).
- Rare classes (PRONE, ON_A_KNEE) remain at 0-17% F1 in both experiments.
- Side-by-side confusion matrix heatmap saved to `work_dir/hockey/gap_v2/comparison_vs_baseline.png`.

---

## TODO 2: Text-to-Joint Alignment Verification (`verify_text_alignment.py`)

### gap_v2 (text-guided, alpha=0.8)

**Command**: `python verify_text_alignment.py --run_dir work_dir/hockey/gap_v2 --split val`

#### 5x5 Alignment Matrix (cosine similarity)

| Skel Part | global text | head text | hand text | hip text | foot text | Best Match |
|-----------|-------------|-----------|-----------|----------|-----------|------------|
| global    | **-0.040**  | -0.018*   | -0.024    | -0.019   | -0.021    | head       |
| head      | -0.005      | -0.000    | **+0.003*** | -0.002 | +0.001    | hand       |
| hand      | -0.009      | +0.008    | +0.009    | +0.006   | **+0.011*** | foot     |
| hip       | -0.008      | +0.007    | **+0.010*** | +0.006 | +0.008    | hand       |
| foot      | -0.005      | +0.000    | **+0.004*** | -0.001 | +0.003    | hand       |

**Diagonal alignment**: 0/5 parts correctly aligned with their own text.

### baseline_v2 (no text guidance, alpha=0.0)

**Command**: `python verify_text_alignment.py --run_dir work_dir/hockey/baseline_v2 --split val`

#### 5x5 Alignment Matrix (cosine similarity)

| Skel Part | global text | head text | hand text | hip text | foot text | Best Match |
|-----------|-------------|-----------|-----------|----------|-----------|------------|
| global    | -0.037      | -0.008    | **+0.004*** | +0.000 | -0.008    | hand       |
| head      | -0.018      | +0.002    | -0.010    | -0.001   | **+0.014*** | foot     |
| hand      | -0.070      | **-0.059*** | -0.063  | -0.073   | -0.077    | head       |
| hip       | -0.074      | -0.060    | -0.053    | -0.052   | **-0.034*** | foot     |
| foot      | **+0.023*** | +0.016   | +0.009    | +0.013   | +0.006    | global     |

**Diagonal alignment**: 0/5 parts correctly aligned with their own text.

### Key Takeaways
- **Neither experiment achieves correct part-to-text alignment.** In both baseline and text-guided, 0 out of 5 parts are best-aligned with their own text description.
- All cosine similarity values are near zero (range: -0.08 to +0.01), indicating the skeleton features and text embeddings live in essentially unrelated subspaces.
- The text-guided model (gap_v2) does *not* show improved diagonal alignment vs baseline, confirming the contrastive loss is non-functional.
- The PRONE class shows 0.000 similarity everywhere (too few samples to produce meaningful features).

---

## Overall Conclusion

These diagnostic tools confirm that the part-aware contrastive learning in the hockey adaptation is **not working as intended**:

1. **No classification benefit**: Text guidance gives +0.08% Top-1 / -0.5% macro F1 (within noise).
2. **No alignment learned**: The 5x5 alignment matrices show random/near-zero cosine similarities with no diagonal dominance in either experiment.
3. **Root cause**: The bugs documented in CLAUDE.md (Bug 1: non-part-specific text prompts, Bug 2: indexing mismatch, Bug 3: hip/foot swap) prevent meaningful contrastive learning from occurring.

Fixing these bugs and retraining is required before the text-guided approach can be properly evaluated.

---

## TODO 1: Per-Part Loss Tracking — Full Training Analysis (`gap_v2_tracked`, 50 epochs)

**Experiment**: `gap_v2_tracked` — same config as `gap_v2` (alpha=0.8), 50 epochs.
**Best val epoch**: 24 (75.00% Top-1) | **Final epoch**: 50

---

### Raw Confusion Matrix (Val, Best Epoch 24)

```
True \ Pred  GLID_FORW ACCEL_FOR GLID_BACK ACCEL_BAC TRANS_F2B TRANS_B2F POST_WHIS   FACEOFF  MAINTAIN     PRONE ON_A_KNEE
GLID_FORW        806 *      23        18         1         3         0         4        30         2         0         0
ACCEL_FORW       115       421 *       1         2         1         1         0         9         2         0         0
GLID_BACK         84         3        52 *       1         0         0         1         3         0         0         0
ACCEL_BACK        12        15         4        23 *       1         2         0         0         0         0         0
TRANS_F2B         16         0         0         1        27 *       0         0         2         0         0         0
TRANS_B2F          5        10         0         3         0        11 *       0         1         0         0         0
POST_WHIS         27         0         0         0         0         0         2 *       0         0         0         0
FACEOFF            7         0         0         0         0         0         0        55 *       2         2         0
MAINTAIN          15         0         1         0         0         0         0        34         1 *       0         0
PRONE              0         0         0         0         0         0         0         0         0         0 *       0
ON_A_KNEE          1         1         0         0         0         0         0         0         0         0         0 *
```

---

### Per-Class TP / FP / FN / TN Breakdown (Val, Best Epoch 24)

| Class       |  TP |  FP |  FN |    TN | Precision | Recall |   F1 | Specificity |  FPR |  FNR | Support |
|-------------|-----|-----|-----|-------|-----------|--------|------|-------------|------|------|---------|
| GLID_FORW   | 806 | 282 |  81 |   695 |     74.1% |  90.9% | 81.6%|       71.1% | 28.9%|  9.1%|     887 |
| ACCEL_FORW  | 421 |  52 | 131 |  1260 |     89.0% |  76.3% | 82.1%|       96.0% |  4.0%| 23.7%|     552 |
| GLID_BACK   |  52 |  24 |  92 |  1696 |     68.4% |  36.1% | 47.3%|       98.6% |  1.4%| 63.9%|     144 |
| ACCEL_BACK  |  23 |   8 |  34 |  1799 |     74.2% |  40.4% | 52.3%|       99.6% |  0.4%| 59.6%|      57 |
| TRANS_F2B   |  27 |   5 |  19 |  1813 |     84.4% |  58.7% | 69.2%|       99.7% |  0.3%| 41.3%|      46 |
| TRANS_B2F   |  11 |   3 |  19 |  1831 |     78.6% |  36.7% | 50.0%|       99.8% |  0.2%| 63.3%|      30 |
| POST_WHIS   |   2 |   5 |  27 |  1830 |     28.6% |   6.9% | 11.1%|       99.7% |  0.3%| 93.1%|      29 |
| FACEOFF     |  55 |  79 |  11 |  1719 |     41.0% |  83.3% | 55.0%|       95.6% |  4.4%| 16.7%|      66 |
| MAINTAIN    |   1 |   6 |  50 |  1807 |     14.3% |   2.0% |  3.4%|       99.7% |  0.3%| 98.0%|      51 |
| PRONE       |   0 |   2 |   0 |  1862 |      0.0% |   0.0% |  0.0%|       99.9% |  0.1%|  0.0%|       0 |
| ON_A_KNEE   |   0 |   0 |   2 |  1862 |      0.0% |   0.0% |  0.0%|      100.0% |  0.0%|100.0%|       2 |
| **MACRO**   |     |     |     |       | **55.3%** |**43.1%**|**45.2%**|        |      |      |         |

**Overall Top-1**: 1398/1864 = **75.00%** | **Mean Class Acc**: 43.12%

---

### TensorBoard: Per-Part Contrastive Loss (Key Epochs, 50-Epoch Run)

| Part   | Ep 1   | Ep 5   | Ep 10  | Ep 20  | Ep 30  | Ep 40  | Ep 50  |
|--------|--------|--------|--------|--------|--------|--------|--------|
| global | 1.2543 | 0.8273 | 0.6653 | 0.5540 | 0.4898 | 0.2702 | 0.0346 |
| head   | 1.2747 | 0.8315 | 0.6666 | 0.5513 | 0.4881 | 0.2675 | 0.0339 |
| hand   | 1.2865 | 0.8372 | 0.6678 | 0.5511 | 0.4884 | 0.2683 | 0.0338 |
| hip    | 1.2685 | 0.8306 | 0.6686 | 0.5519 | 0.4904 | 0.2675 | 0.0336 |
| foot   | 1.1829 | 0.8309 | 0.6687 | 0.5524 | 0.4885 | 0.2685 | 0.0339 |

### TensorBoard: Per-Part Cosine Similarity (Key Epochs, 50-Epoch Run)

| Part   | Ep 1  | Ep 5  | Ep 10 | Ep 20 | Ep 30 | Ep 40 | Ep 50 |
|--------|-------|-------|-------|-------|-------|-------|-------|
| global | 0.113 | 0.430 | 0.422 | 0.452 | 0.482 | 0.629 | 0.962 |
| head   | 0.118 | 0.427 | 0.411 | 0.455 | 0.486 | 0.636 | 0.964 |
| hand   | 0.095 | 0.441 | 0.424 | 0.451 | 0.480 | 0.632 | 0.961 |
| hip    | 0.122 | 0.426 | 0.412 | 0.451 | 0.482 | 0.634 | 0.964 |
| foot   | 0.225 | 0.449 | 0.423 | 0.451 | 0.486 | 0.629 | 0.962 |

### Cross-Part Correlation (step-level, all 50 epochs)

| Pair           | Correlation | Mean Abs Diff |
|----------------|-------------|---------------|
| head vs global | 0.994       | 0.015         |
| head vs hand   | 0.996       | 0.013         |
| head vs hip    | 0.997       | 0.013         |
| head vs foot   | 0.995       | 0.016         |

Mean spread between parts per step: **0.12 → 0.003** (collapsed to near-zero by epoch 50).

### CE Loss by Epoch (Key Checkpoints)

| Ep 1   | Ep 5   | Ep 10  | Ep 20  | Ep 30  | Ep 40  | Ep 50  |
|--------|--------|--------|--------|--------|--------|--------|
| 1.5765 | 0.8757 | 0.7045 | 0.5860 | 0.5104 | 0.2551 | 0.0131 |

---

### Key Takeaways

**Confusion matrix (best epoch 24, 75.00% Top-1):**
- GLID_FORW (91% recall) and ACCEL_FORW (76% recall) are strong but GLID_FORW still carries 282 FPs (FPR=29%) — absorbing backward-skating variants.
- FACEOFF has asymmetric performance: high recall (83%) but low precision (41%) due to MAINTAIN (34 FPs) and GLID_FORW (30 FPs) being predicted as FACEOFF.
- TRANS_F2B is the best-performing minority class (58.7% recall, 84.4% precision, F1=69.2%).
- POST_WHIS (6.9% recall) and MAINTAIN (2.0% recall) are nearly collapsed — both look visually similar to forward gliding.
- PRONE has zero val samples; ON_A_KNEE has only 2 — both unlearnable at this dataset size.

**Contrastive loss / cosine sim (TensorBoard, 50 epochs):**
- All 5 part losses remain essentially identical throughout (cross-part correlation >0.994 over all 50 epochs).
- Spread between parts shrinks from 0.12 → 0.003, confirming all losses collapse to a single signal.
- Cosine similarities plateau at ~0.43–0.51 for epochs 1–39, then surge to ~0.96 after the LR step at epoch 40.
- The post-epoch-40 surge is **not alignment** — it is the text encoder (trained at full LR=0.01 after the step) collapsing to reproduce the skeleton feature distribution, since all parts receive the same generic text signal.
- CE loss drops to 0.013 by epoch 50 (heavy overfitting after LR step), which is why best val acc is at epoch 24, not epoch 50.
- **Root cause confirmed at scale**: even over 50 epochs, 0/5 parts learn part-specific alignment. Bug 1 renders the contrastive signal structurally incapable of producing differentiated part learning.
