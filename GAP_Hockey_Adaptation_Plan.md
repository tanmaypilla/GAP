# GAP-based Text-Guided Skeleton Action Recognition for Hockey Skating Actions
## Comprehensive Adaptation Plan

### Executive Summary
This document outlines a plan to adapt the GAP (Generative Action-description Prompts) method for skeleton-based action recognition to classify hockey skating actions. The GAP approach uses language models to generate text descriptions of body part movements and employs CLIP's text encoder to supervise skeleton encoder training through contrastive learning.

---

## 1. Understanding the GAP Method

### 1.1 Core Architecture
- **Skeleton Encoder**: CTR-GCN (Channel-wise Topology Refinement Graph Convolutional Network)
  - Extracts features from skeleton sequences
  - Processes 4 body parts separately: head, hands, hips, feet
  - Outputs 256-dim features per part, projected to 512-dim for text alignment

- **Text Encoder**: CLIP (ViT-B/32 or other variants)
  - Encodes text descriptions into feature vectors
  - Pre-trained on large-scale image-text pairs

- **Multi-modal Alignment**:
  - Uses KL divergence loss to align skeleton features with text features
  - Contrastive learning with temperature scaling
  - 5 different text prompt variations per action class

### 1.2 Text Prompt Generation Strategy
The original GAP uses:
1. **Synonym prompts**: Alternative names for actions
2. **Sentence prompts**: Full sentence descriptions
3. **PASTA prompts**: Part-Aware Structured Text Augmentation
   - Describes movements of different body parts separately
   - Format: `[action name], [head movement], [hand movement], [hip movement], [foot movement], [additional details]`

### 1.3 Training Process
1. Generate text embeddings for all action classes using CLIP
2. Extract skeleton features from 4 body parts
3. Project skeleton features to match text embedding dimension
4. Compute KL divergence loss between skeleton and text features
5. Combine with standard classification loss

---

## 2. Dataset Analysis

### 2.1 Hockey Skating Actions Dataset
- **Total Actions**: 32,199 clips
- **Classes**: 12 skating actions
  - GLID_FORW (Gliding Forward)
  - ACCEL_FORW (Accelerating Forward)
  - GLID_BACK (Gliding Backward)
  - ACCEL_BACK (Accelerating Backward)
  - TRANS_FORW_TO_BACK (Transition Forward to Back)
  - TRANS_BACK_TO_FORW (Transition Back to Forward)
  - POST_WHISTLE_GLIDING (Post-Whistle Gliding)
  - RAPID_DECELERATION (Rapid Deceleration)
  - FACEOFF_BODY_POSITION (Faceoff Body Position)
  - MAINTAIN_POSITION (Maintain Position)
  - PRONE (Prone Position)
  - ON_A_KNEE (On a Knee)

- **Clip Duration**: 30 frames (1 second at 30 fps)
- **Skeleton Format**: 2D pose keypoints
- **Data Structure**: 
  - `annotations/train.pkl`, `val.pkl`, `test.pkl`
  - `actions.json` with metadata
  - `annotations_with_pose.json` with pose data

### 2.2 Key Differences from NTU RGB+D
- **Skeleton Structure**: Need to verify if using same 25-joint format or different
- **Temporal Length**: 30 frames vs 64 frames (need padding/truncation strategy)
- **Action Domain**: Sports-specific vs general daily actions
- **2D vs 3D**: Hockey uses 2D keypoints vs NTU's 3D

---

## 3. Adaptation Strategy

### 3.1 Phase 1: Data Preparation and Exploration

#### 3.1.1 Dataset Structure Analysis
**Tasks**:
1. Load and inspect pickle files to understand data format
2. Verify skeleton keypoint structure (number of joints, format)
3. Check if skeleton follows standard pose estimation format (e.g., COCO, OpenPose)
4. Analyze temporal distribution of actions
5. Check class balance in train/val/test splits

**Deliverables**:
- Data exploration notebook
- Dataset statistics report
- Skeleton format documentation

#### 3.1.2 Data Preprocessing Pipeline
**Tasks**:
1. Create data loader compatible with GAP framework
2. Handle 30-frame sequences (pad to 64 or use different window size)
3. Normalize skeleton coordinates
4. Convert to format: `(N, C, T, V, M)` where:
   - N: batch size
   - C: channels (x, y coordinates = 2)
   - T: temporal frames (30 or padded to 64)
   - V: number of joints
   - M: number of persons (typically 1 for hockey)

**Deliverables**:
- `feeders/feeder_hockey.py` - Custom data feeder
- Data preprocessing scripts
- Unit tests for data loading

### 3.2 Phase 2: Text Prompt Generation

#### 3.2.1 Action Description Generation
**Strategy**: Use LLM (GPT-4/Claude) to generate rich descriptions following GAP's PASTA format

**For each action class, generate**:
1. **Base action name**: Short descriptive name
2. **Synonym variations**: 3-5 alternative names
3. **Full sentence descriptions**: 2-3 complete sentences
4. **PASTA format descriptions**: Structured by body parts
   - Head/upper body movement
   - Hand/arm movement  
   - Hip/torso movement
   - Foot/leg movement
   - Additional context

**Example for GLID_FORW**:
```
Base: "gliding forward"
Synonyms: "forward glide", "skating forward", "moving forward on ice"
Sentences: 
  - "The player glides forward maintaining balance"
  - "Smooth forward movement on ice surface"
PASTA:
  - "gliding forward, head forward and balanced, arms extended for balance, torso slightly forward, legs alternating in forward motion, maintaining momentum on ice"
```

**Tasks**:
1. Generate text prompts for all 12 classes
2. Create multiple variations (5 per class as in GAP)
3. Save in format compatible with `Text_Prompt.py`

**Deliverables**:
- `text/hockey_label_map.txt` - Action class names
- `text/hockey_synonym_openai_t01.txt` - Synonym variations
- `text/hockey_sentence_openai_t01.txt` - Sentence descriptions
- `text/hockey_pasta_openai_t01.txt` - PASTA format descriptions

#### 3.2.2 Text Prompt Integration
**Tasks**:
1. Add functions to `Text_Prompt.py` for hockey dataset
2. Create `text_prompt_openai_pasta_pool_4part_hockey()` function
3. Ensure compatibility with existing CLIP tokenization

**Deliverables**:
- Updated `Text_Prompt.py` with hockey-specific functions

### 3.3 Phase 3: Model Adaptation

#### 3.3.1 Graph Structure Adaptation
**Tasks**:
1. Determine skeleton joint structure used in hockey dataset
2. If different from NTU's 25 joints, create new graph definition
3. Map joints to body parts (head, hands, hips, feet)
4. Define adjacency matrix for graph convolution

**Considerations**:
- If using COCO format (17 joints): Need to adapt
- If using OpenPose format (25 joints): May be compatible
- If custom format: Need to create new graph structure

**Deliverables**:
- `graph/hockey.py` - Graph definition for hockey skeleton
- Updated body part mappings in model

#### 3.3.2 Model Configuration
**Tasks**:
1. Adapt `Model_lst_4part` for hockey dataset
2. Adjust input dimensions:
   - `num_point`: Number of skeleton joints
   - `num_person`: Typically 1 for hockey
   - `in_channels`: 2 for 2D (x, y) vs 3 for 3D
3. Modify body part extraction if joint indices differ
4. Adjust temporal processing for 30-frame sequences

**Deliverables**:
- `model/ctrgcn.py` - Add `Model_lst_4part_hockey` class
- Configuration files in `config/hockey/`

#### 3.3.3 Loss Function
**Tasks**:
1. Verify KL divergence loss implementation (`KLLoss.py`)
2. Ensure temperature scaling works correctly
3. Combine with classification loss
4. Test with hockey data dimensions

**No changes needed** - Existing `KLLoss.py` should work

### 3.4 Phase 4: Configuration and Training Setup

#### 3.4.1 Configuration Files
**Tasks**:
1. Create `config/hockey/default.yaml`
2. Set dataset paths
3. Configure model parameters
4. Set training hyperparameters:
   - Learning rate: Start with 0.1 (as in NTU)
   - Batch size: Adjust based on GPU memory
   - Epochs: Start with 65 (as in NTU)
   - Weight decay: 0.0004

**Deliverables**:
- `config/hockey/default.yaml`
- `config/hockey/lst_joint.yaml` (joint modality)
- `config/hockey/lst_bone.yaml` (bone modality, optional)
- `config/hockey/lst_joint_vel.yaml` (velocity modality, optional)

#### 3.4.2 Training Script
**Tasks**:
1. Create `main_multipart_hockey.py` based on `main_multipart_ntu.py`
2. Adapt data loading
3. Integrate hockey-specific text prompts
4. Set up logging and checkpointing

**Deliverables**:
- `main_multipart_hockey.py` - Main training script

### 3.5 Phase 5: Training and Evaluation

#### 3.5.1 Initial Training
**Tasks**:
1. Run baseline training without text supervision
2. Establish baseline accuracy
3. Compare with text-guided training

**Metrics to Track**:
- Top-1 accuracy
- Top-5 accuracy (if applicable)
- Per-class accuracy
- Confusion matrix
- Training/validation loss curves

#### 3.5.2 Text-Guided Training
**Tasks**:
1. Train with GAP method
2. Monitor text-skeleton alignment loss
3. Compare performance with baseline
4. Analyze which text prompts work best

#### 3.5.3 Ablation Studies
**Tasks**:
1. Test different text prompt variations
2. Test with/without different body parts
3. Test different CLIP model variants
4. Test different loss weightings

---

## 4. Implementation Checklist

### Phase 1: Data Preparation
- [X] Explore dataset structure and format
- [X] Document skeleton keypoint format
- [X] Create data preprocessing pipeline
- [X] Implement `feeder_hockey.py`
- [X] Test data loading and verify format compatibility

### Phase 2: Text Prompts
- [X] Generate action descriptions using LLM
- [X] Create synonym variations
- [X] Create sentence descriptions
- [X] Create PASTA format descriptions
- [X] Integrate into `Text_Prompt.py`
- [X] Test CLIP tokenization

### Phase 3: Model Adaptation
- [X] Determine skeleton graph structure
- [X] Create/adapt graph definition
- [X] Create hockey-specific model class
- [X] Adapt body part extraction
- [X] Test model forward pass

### Phase 4: Configuration
- [X] Create configuration files
- [X] Set up training script
- [X] Configure logging and checkpoints
- [X] Test configuration loading

### Phase 5: Training
- [X] Run baseline training
- [X] Run GAP training
- [X] Evaluate and compare results
- [ ] Perform ablation studies
- [ ] Document findings

---

## 5. Expected Challenges and Solutions

### Challenge 1: Skeleton Format Mismatch
**Problem**: Hockey dataset may use different skeleton format than NTU RGB+D
**Solution**: 
- Create custom graph definition
- Map joints to standard body parts
- Adapt model input processing

### Challenge 2: Shorter Temporal Sequences
**Problem**: 30 frames vs 64 frames in NTU
**Solution**:
- Option A: Pad sequences to 64 frames
- Option B: Adapt model to handle 30 frames directly
- Option C: Use different temporal window size

### Challenge 3: 2D vs 3D Coordinates
**Problem**: Hockey uses 2D keypoints, GAP was designed for 3D
**Solution**:
- Set `in_channels=2` instead of 3
- Model architecture should handle this automatically
- May need to adjust normalization

### Challenge 4: Domain-Specific Actions
**Problem**: Hockey actions are very specific and may not have rich text descriptions
**Solution**:
- Use domain knowledge to create detailed descriptions
- Focus on body part movements specific to skating
- Consider consulting hockey experts for descriptions

### Challenge 5: Class Imbalance
**Problem**: Some actions may be more common than others
**Solution**:
- Analyze class distribution
- Use weighted loss if needed
- Consider data augmentation for minority classes

---

## 6. Success Metrics

### Primary Metrics
- **Top-1 Accuracy**: Target >85% (baseline comparison)
- **Improvement over baseline**: Target +2-5% improvement with GAP
- **Per-class accuracy**: All classes >70%

### Secondary Metrics
- Training stability (loss convergence)
- Text-skeleton alignment quality
- Generalization to test set
- Inference speed (should be same as baseline)

---

## 7. Timeline Estimate

- **Phase 1 (Data Prep)**: 3-5 days
- **Phase 2 (Text Prompts)**: 2-3 days
- **Phase 3 (Model Adaptation)**: 5-7 days
- **Phase 4 (Configuration)**: 2-3 days
- **Phase 5 (Training & Eval)**: 7-10 days

**Total**: ~3-4 weeks for complete implementation and initial results

---

## 8. Next Steps

1. **Immediate**: Start with Phase 1 - explore dataset structure
2. **Week 1**: Complete data preparation and text prompt generation
3. **Week 2**: Implement model adaptations and configurations
4. **Week 3**: Begin training and evaluation
5. **Week 4**: Refinement and documentation

---

## 9. References

- GAP Paper: "Generative Action Description Prompts for Skeleton-based Action Recognition" (ICCV 2023)
- CTR-GCN: Channel-wise Topology Refinement Graph Convolutional Network
- CLIP: Learning Transferable Visual Models From Natural Language Supervision
- Original GAP Repository: `/home/tanmay-ura/GAP`

---

## Appendix: File Structure

```
GAP/
├── config/
│   └── hockey/
│       ├── default.yaml
│       ├── lst_joint.yaml
│       └── lst_bone.yaml
├── feeders/
│   └── feeder_hockey.py
├── graph/
│   └── hockey.py
├── model/
│   └── ctrgcn.py (updated)
├── text/
│   ├── hockey_label_map.txt
│   ├── hockey_synonym_openai_t01.txt
│   ├── hockey_sentence_openai_t01.txt
│   └── hockey_pasta_openai_t01.txt
├── Text_Prompt.py (updated)
└── main_multipart_hockey.py
```

---

**Document Version**: 1.0  
**Date**: 2024  
**Author**: Adaptation Plan for Hockey Skating Actions

**TODO**:
28th jan
- Improve Text Prompt Generation 
- Use Top 2 or 3 instead of Top 5 for a 12-class problem
- Check for differences between per-class accuracy when using text and no text supervision
- Feed it unrelated prompts to see if it can generalize to new actions (for training) (ok maybe not useful)

4th feb
- Calculate for Mean class accuracy (overall) and per-class accuracy (overall)
- Use the new dataset with one less class (Rapid Deceleration)
- Use new text prompts from Prof
