# GAP Hockey Adaptation — Code Analysis

## Project Overview

GAP (Generative Action-description Prompts) is a skeleton-based action recognition framework that uses CLIP text encoders to provide part-aware semantic supervision during training. The core idea: align body-part skeleton features with text descriptions of those body parts via contrastive learning (KL divergence), so the skeleton encoder learns richer representations.

This codebase adapts GAP from NTU RGB+D (120 classes, 25 joints, 3D) to hockey skating actions (11 classes, 20 joints, 2D).

---

## Directory Structure

```
GAP/
├── main_multipart_hockey.py        # Training script (671 lines) — MODIFIED from NTU
├── main_multipart_ntu.py           # Original NTU training script (reference)
├── main_multipart_ucla.py          # UCLA training script (reference)
│
├── model/
│   ├── ctrgcn.py                   # Model definitions — contains Model_lst_4part_hockey (line 883)
│   ├── baseline.py                 # TextCLIP wrapper for CLIP text encoder
│   ├── activation.py               # Activation functions
│   └── utils/                      # GCN building blocks (tgcn.py, graph.py)
│
├── feeders/
│   ├── feeder_hockey.py            # Hockey dataset loader (142 lines) — NEW
│   ├── feeder_ntu.py               # NTU dataset loader (reference)
│   ├── feeder_ucla.py              # UCLA dataset loader
│   ├── bone_pairs.py               # Bone pair definitions
│   └── tools.py                    # Data augmentation utilities
│
├── graph/
│   ├── hockey.py                   # 20-point hockey skeleton graph — NEW
│   ├── ntu_rgb_d.py                # 25-point NTU skeleton (reference)
│   ├── tools.py                    # Graph normalization (get_spatial_graph, get_adjacency_matrix)
│   └── infogcn/                    # InfoGCN graph variants
│
├── text/
│   ├── hockey_label_map.txt        # 11 hockey class names (v2, no RAPID_DECELERATION)
│   ├── hockey_synonym_gemini_t01.txt   # Synonym prompts (Gemini-generated)
│   ├── hockey_sentence_gemini_t01.txt  # Sentence descriptions
│   ├── hockey_pasta_gemini_t01.txt     # PASTA part-aware descriptions
│   ├── pasta_openai_t01.txt            # NTU PASTA text (reference)
│   └── synonym_openai_t01.txt          # NTU synonyms (reference)
│
├── config/hockey/
│   ├── default.yaml                # Baseline joint config
│   ├── lst_joint.yaml              # LST joint variant
│   ├── lst_bone.yaml               # Bone modality
│   ├── lst_joint_vel.yaml          # Joint velocity
│   ├── lst_bone_vel.yaml           # Bone velocity
│   └── vit_large.yaml              # ViT-L/14 text encoder
│
├── work_dir/hockey/                # Training outputs
│   ├── baseline_joint/             # loss_alpha=0.0, 65 epochs → 78.24% Top-1
│   ├── text_guided_0.8/            # loss_alpha=0.8, 65 epochs → 78.41% Top-1
│   ├── ablation_alpha_0.4/         # loss_alpha=0.4, 65 epochs → 78.83% Top-1
│   ├── baseline_weighted_fix/      # loss_alpha=0.0, weighted CE, 45 epochs → 65.91% Top-1
│   ├── text_guided_0.8_weighted/   # loss_alpha=0.8, weighted CE, 45 epochs → 74.74% Top-1
│   └── ablation_vit_large/         # ViT-L/14, early stopped
│
├── Text_Prompt.py                  # Text prompt loaders — contains text_prompt_hockey() (line 189)
├── KLLoss.py                       # KL divergence loss for contrastive alignment
├── tools.py                        # gen_label(), create_logits(), convert_models_to_fp32()
├── clip/                           # CLIP model (clip.py, model.py, simple_tokenizer.py)
├── ensemble.py                     # Multi-modality ensemble
├── plot_logs.py                    # Training curve visualization
├── eval_matrix.py                  # Confusion matrix evaluation
├── calculate_weights.py            # Class weight calculator
├── analysis/                       # Dataset exploration scripts
└── torchlight/                     # Training framework utilities
```

---

## Architecture Flow

```
Input: (N, 2, 64, 20, 1)  — batch, xy-coords, time, joints, persons
    ↓
[Data BatchNorm] → normalize across spatial dims
    ↓
[A_vector transform] → apply graph adjacency to input features
    ↓
[10 TCN-GCN blocks] → l1-l10, channels: 64→64→64→64→128→128→128→256→256→256
    ↓ stride=2 at l5 and l8 (temporal downsampling: 64→32→16)
    ↓
    ├── [Global Feature] → mean over (V,T,M) → Linear(256→512) → feature_dict
    │      ↓
    │   [Contrastive Loss ind=0] ← align with global text embedding
    │
    └── [4 Part Features] → select joint subsets → mean over (V,T,M) → Linear(256→512)
           ├── head_feature  (joints 0,1,2)      → [Loss ind=1] ← should align with HEAD text
           ├── hand_feature  (joints 3,4,7-10,17-19) → [Loss ind=2] ← should align with HAND text
           ├── hip_feature   (joints 5,6)         → [Loss ind=3] ← should align with HIP text
           └── foot_feature  (joints 11-16)       → [Loss ind=4] ← should align with FOOT text
    ↓
[Classification Head] → Linear(256→11) → CrossEntropy loss
    ↓
Total Loss = CE_loss + alpha * mean(contrastive_losses)
```

---

## Hockey Dataset

- **11 Classes** (v2 — RAPID_DECELERATION removed): GLID_FORW, ACCEL_FORW, GLID_BACK, ACCEL_BACK, TRANS_FORW_TO_BACK, TRANS_BACK_TO_FORW, POST_WHISTLE_GLIDING, FACEOFF_BODY_POSITION, MAINTAIN_POSITION, PRONE, ON_A_KNEE
- **20 Keypoints**: 0-2 (head), 3-4 (shoulders), 5-6 (hips), 7-10 (arms), 11-16 (legs/feet), 17-19 (stick)
- **Format**: 2D (x,y) coordinates, ~30 frames per clip, single person
- **Preprocessing**: Hip-center normalization, scale by 1/1000, temporal resize to 64 frames

---

## Training Hyperparameters (default config)

| Parameter | Value |
|---|---|
| base_lr | 0.1 |
| optimizer | SGD + Nesterov |
| weight_decay | 0.0004 |
| batch_size | 64 |
| LR schedule | step decay at [35, 55] (or [30, 40] for 45-epoch runs) |
| lr_decay_rate | 0.1 |
| warm_up_epoch | 0 |
| loss_alpha | 0.8 (text weight) or 0.0 (baseline) |
| te_lr_ratio | 1.0 |
| CLIP model | ViT-B/32 |

---

## Training Results

| Experiment | Top-1 (Final) | Top-1 (Peak) | Epochs | Alpha | Weighted CE |
|---|---|---|---|---|---|
| baseline_joint | 77.97% | ~78.24% | 65 | 0.0 | No |
| text_guided_0.8 | 78.36% | ~78.41% | 65 | 0.8 | No |
| ablation_alpha_0.4 | 78.26% | ~78.83% | 65 | 0.4 | No |
| baseline_weighted_fix | 65.69% | ~65.91% | 45 | 0.0 | Yes |
| text_guided_0.8_weighted | 74.28% | ~74.74% | 45 | 0.8 | Yes |

**Conclusion**: Text guidance provides <1% improvement (within noise). Weighted CE hurts badly. The text-guided loss is essentially non-functional due to bugs documented below.

---

## Critical Bugs

### BUG 1 (CRITICAL): Text Prompts Are NOT Part-Specific

**Location**: `Text_Prompt.py:189-244` and `main_multipart_hockey.py:48-77`

**The Problem**: GAP's part-aware contrastive learning depends on aligning each body part feature with text describing THAT specific body part. The NTU implementation does this correctly; the hockey implementation does not.

**NTU approach** (`text_prompt_openai_pasta_pool_4part()` in `Text_Prompt.py:105-125`):

The NTU PASTA text format has 7 semicolon-delimited segments per class:
```
drink water;head tilts back slightly; hand grasps cup; arm lifts cup to mouth; hip remains stationary; leg remains stationary; foot remains stationary.
```

The function splits these and creates per-part text dictionaries:
- `text_dict[0]` = action name only → global contrastive (all classes, shape [num_classes, 77])
- `text_dict[1]` = action + head desc → aligned with `head_feature`
- `text_dict[2]` = action + hand/arm desc → aligned with `hand_feature`
- `text_dict[3]` = action + hip desc → aligned with `hip_feature`
- `text_dict[4]` = action + leg/foot desc → aligned with `foot_feature`

**Hockey approach** (`text_prompt_hockey()` in `Text_Prompt.py:189-244`):

Returns a flat list of different prompt TYPES per class (not body parts):
```python
class_prompts = [
    "A video of a person glid forw.",       # ind=0: label template 1
    "A hockey player glid forw.",           # ind=1: label template 2
    "forward gliding, coasting on ice...",  # ind=2: synonym
    "The player coasts forward...",         # ind=3: sentence
    "Head looks forward; Hands hold...",    # ind=4: FULL PASTA (all parts combined)
]
```

In the training loop (`main_multipart_hockey.py:486-516`), these get used as:
- ind=0 → random text → global contrastive (OK)
- ind=1 → "A hockey player {label}" → aligned with `head_feature` (WRONG — not head-specific)
- ind=2 → synonym text → aligned with `hand_feature` (WRONG — not hand-specific)
- ind=3 → sentence desc → aligned with `hip_feature` (WRONG — not hip-specific)
- ind=4 → full PASTA line → aligned with `foot_feature` (WRONG — contains all parts)

**Impact**: The part-aware contrastive loss is semantically meaningless. Every body part trains against generic action text rather than part-specific descriptions. This is why text guidance provides zero benefit.

**Fix Required**: Create a hockey equivalent of `text_prompt_openai_pasta_pool_4part()` that:
1. Splits the hockey PASTA text (`hockey_pasta_gemini_t01.txt`) by semicolons into 4 segments (Head, Hands, Hips, Legs)
2. Constructs `text_dict[aug_id]` = tensor of shape `[num_classes, 77]` (matching NTU's structure)
3. Maps: `text_dict[1]` = head text, `text_dict[2]` = hand text, `text_dict[3]` = hip text, `text_dict[4]` = foot text

Also update `main_multipart_hockey.py` lines 48-77 to use the new function and restructure `text_dict` indexing.

---

### BUG 2 (HIGH): text_dict Indexing Order Reversed

**Location**: `main_multipart_hockey.py:491` vs `main_multipart_ntu.py:453`

```python
# Hockey (class-first indexing):
texts = torch.stack([text_dict[int(i)][int(j)] for i,j in zip(label, text_id)])
#                    text_dict[class][prompt_type]

# NTU (prompt-type-first indexing):
texts = torch.stack([text_dict[j][i,:] for i,j in zip(label, text_id)])
#                    text_dict[prompt_type][class, :]
```

The data structures differ:
- Hockey `text_dict`: `{class_id: tensor[num_prompts, 77]}`
- NTU `text_dict`: `{prompt_type_id: tensor[num_classes, 77]}`

This is architecturally consistent with each codebase's text loading, but it means fixing Bug 1 requires also restructuring the text_dict to match NTU's format and updating the indexing in the training loop.

---

### BUG 3 (HIGH): Hip/Foot Feature Variable Swap

**Location**: `model/ctrgcn.py:986-987`

```python
# Hockey (WRONG — names swapped relative to joint lists):
hip_feature  = self.part_list[2](feature[:,:,:,:,foot_list].mean(4).mean(3).mean(1))  # foot joints!
foot_feature = self.part_list[3](feature[:,:,:,:,hip_list].mean(4).mean(3).mean(1))   # hip joints!

# NTU (CORRECT — names match joint lists):
foot_feature = self.part_list[2](feature[:,:,:,:,foot_list].mean(4).mean(3).mean(1))
hip_feature  = self.part_list[3](feature[:,:,:,:,hip_list].mean(4).mean(3).mean(1))
```

The return statement is `[head_feature, hand_feature, hip_feature, foot_feature]`, so:
- Index 2 (`hip_feature`) is computed from `foot_list` joints (knees 11-12, ankles 13-14, feet 15-16)
- Index 3 (`foot_feature`) is computed from `hip_list` joints (hips 5-6)

This means the hip text (if Bug 1 were fixed) would align with leg/foot skeleton features, and vice versa. Even the classification head's global feature (which uses all joints) is unaffected by this since it's averaged globally, but the part-aware losses get crossed signals.

**Fix**: Swap the joint lists on lines 986-987:
```python
hip_feature  = self.part_list[2](feature[:,:,:,:,hip_list].mean(4).mean(3).mean(1))
foot_feature = self.part_list[3](feature[:,:,:,:,foot_list].mean(4).mean(3).mean(1))
```

---

### BUG 4 (MEDIUM): Disconnected Stick Nodes in GCN

**Location**: `graph/hockey.py:41-43` and `model/ctrgcn.py:975`

The hockey stick nodes (17, 18, 19) form a disconnected subgraph — no edges connect them to any body joint. Through 10 GCN message-passing layers, stick node features never receive information from body nodes (and vice versa). After the GCN, stick features are essentially random/uninformed.

Yet these nodes are assigned to `hand_list` at line 975:
```python
hand_list = torch.Tensor([3, 4, 7, 8, 9, 10, 17, 18, 19]).long()
```

This means 3 of the 9 joints averaged into `hand_feature` carry uninformative features, diluting the signal.

**Options**:
1. Add an edge from the nearest wrist to stick node 17 (e.g., `(10, 17)` or `(8, 17)`)
2. Remove stick nodes from `hand_list`
3. Both: connect stick to body AND keep in hand_list (recommended if stick position is meaningful)

---

### BUG 5 (LOW): Unnecessary Temporal Upsampling

**Location**: `feeders/feeder_hockey.py:131-142`

Raw data is 30 frames (1 second at 30fps). Linear interpolation to 64 frames creates synthetic intermediate frames without adding information. This doubles computation and may introduce smoothing artifacts.

**Options**:
1. Set `window_size=30` in configs and adjust model (temporal dim changes from 64→30, after two stride-2 layers: 30→15→7)
2. Use `window_size=32` (nearest power of 2, gives clean 32→16→8)
3. Keep 64 but use padding instead of interpolation

---

### ISSUE 6 (MEDIUM): Weighted Cross-Entropy Too Aggressive

**Location**: `main_multipart_hockey.py:337-347`

```python
weights_list = [
    0.192, 0.244, 1.144, 4.671, 3.776, 3.751,
    4.202, 9.419, 2.442, 3.879, 26.786, 43.957
]
```

Classes 10 (PRONE) and 11 (ON_A_KNEE) have weights 26.8x and 44.0x the dominant class. These extreme weights amplify noisy gradients from rare-class samples and destabilize training. The weighted experiments show 12-13% accuracy drops versus unweighted.

**Fix**: Use square-root or log-scaled weights, or cap weights at 5-10x. Example:
```python
weights_list = [w**0.5 for w in weights_list]  # square root dampening
```

---

### ISSUE 7 (LOW): Text Encoder Learning Rate

**Location**: `main_multipart_hockey.py:396-397`

```python
{'params': self.model_text_dict.parameters(), 'lr': self.arg.base_lr * self.arg.te_lr_ratio}
```

With `te_lr_ratio=1` and `base_lr=0.1`, the frozen CLIP text encoder is updated at the same learning rate as the skeleton model. CLIP's text encoder has pre-trained weights optimized for image-text alignment. Updating at LR=0.1 risks catastrophic forgetting of CLIP's semantic representations.

**Fix**: Set `te_lr_ratio` to 0.01 or 0.001, or freeze the text encoder entirely (common practice in GAP).

---

## Key Differences: Hockey vs NTU Implementation

| Aspect | NTU (Original) | Hockey (Adapted) | Issue? |
|---|---|---|---|
| Text loading | `text_prompt_openai_pasta_pool_4part()` → per-part text | `text_prompt_hockey()` → flat prompt list | YES — Bug 1 |
| text_dict format | `{aug_id: tensor[classes, 77]}` | `{class_id: tensor[prompts, 77]}` | YES — Bug 2 |
| Part feature order | foot=part_list[2], hip=part_list[3] | hip=part_list[2](foot_joints), foot=part_list[3](hip_joints) | YES — Bug 3 |
| Skeleton dimensions | 3D (x,y,z), 25 joints | 2D (x,y), 20 joints | OK |
| Temporal frames | 64 native | 30→64 interpolated | Suboptimal |
| CLIP text encoder | Frozen or low LR | Full LR (te_lr_ratio=1) | Risky |
| Number of classes | 60 or 120 | 11 | OK |
| PASTA format | 7 semicolon parts | 4 semicolon parts (not split) | YES — Bug 1 |
| Graph connectivity | Fully connected body | Stick disconnected from body | Bug 4 |
| Loss weighting | Unweighted CE | Weighted CE (up to 44x) | Too aggressive |

---

## Files Modified/Added for Hockey Adaptation

### New Files
- `main_multipart_hockey.py` — Training script (adapted from main_multipart_ntu.py)
- `feeders/feeder_hockey.py` — Dataset loader
- `graph/hockey.py` — 20-point skeleton graph
- `config/hockey/*.yaml` — 6 configuration files
- `text/hockey_*.txt` — 4 text prompt files
- `analysis/*.py` — Dataset exploration
- `test_*.py` — 4 test scripts
- `plot_logs.py`, `eval_matrix.py`, `calculate_weights.py` — Utilities

### Modified Files
- `Text_Prompt.py` — Added `text_prompt_hockey()` function (line 189)
- `model/ctrgcn.py` — Added `Model_lst_4part_hockey` class (line 883) and `Model_lst_4part_bone_hockey` (line 1002)
- `graph/tools.py` — Added `get_adjacency_matrix()` helper

---

## Summary of Recommended Fixes (Priority Order)

1. **Rewrite text prompt loading** to create per-part text embeddings matching NTU's structure. Split hockey PASTA text by semicolons. Restructure `text_dict` to `{aug_id: tensor[num_classes, 77]}`.
2. **Fix hip/foot swap** in `ctrgcn.py:986-987`.
3. **Connect stick nodes** to body skeleton (add edge from wrist to stick top).
4. **Dampen class weights** (square-root or log scaling, or cap at 10x).
5. **Lower text encoder LR** (`te_lr_ratio=0.01` or freeze entirely).
6. **Consider native temporal resolution** (`window_size=32` with padding).
