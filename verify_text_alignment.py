#!/usr/bin/env python
"""
Verify text-to-body-joint alignment by computing a 5x5 cosine similarity matrix
between skeleton part features and text part embeddings.

Loads a trained checkpoint, runs inference on the val/test set, collects per-sample
part features (global + 4 body parts), encodes all 5 text sets via CLIP, and
computes the alignment matrix.

Usage:
    python verify_text_alignment.py --run_dir work_dir/hockey/gap_v2 [--epoch N] [--split val]
"""

import argparse
import os
import re
import glob
from collections import OrderedDict

import numpy as np
import torch
import yaml
from tqdm import tqdm

import clip
from model.baseline import TextCLIP
from Text_Prompt import text_prompt_hockey_pasta_pool_4part


PART_NAMES = ['global', 'head', 'hand', 'hip', 'foot']

CLASS_NAMES = [
    "GLID_FORW", "ACCEL_FORW", "GLID_BACK", "ACCEL_BACK",
    "TRANS_FORW_TO_BACK", "TRANS_BACK_TO_FORW", "POST_WHISTLE_GLIDING",
    "FACEOFF_BODY_POSITION", "MAINTAIN_POSITION", "PRONE", "ON_A_KNEE",
]


def parse_epoch(filename):
    m = re.search(r'runs-(\d+)-', filename)
    return int(m.group(1)) if m else -1


def find_checkpoint(run_dir, epoch=None):
    """Find checkpoint .pt file. If epoch is None, pick the latest."""
    pt_files = glob.glob(os.path.join(run_dir, '*.pt'))
    if not pt_files:
        raise FileNotFoundError(f"No .pt files in {run_dir}")
    if epoch is not None:
        matches = [f for f in pt_files if f'runs-{epoch}-' in os.path.basename(f)]
        if not matches:
            raise FileNotFoundError(f"No checkpoint for epoch {epoch} in {run_dir}")
        return matches[0]
    pt_files.sort(key=lambda f: parse_epoch(os.path.basename(f)))
    return pt_files[-1]


def load_model_and_data(run_dir, epoch, split, device):
    """Load model from checkpoint and create data loader."""
    # Load config
    config_path = os.path.join(run_dir, 'config.yaml')
    if os.path.exists(config_path):
        with open(config_path, 'r') as f:
            config = yaml.safe_load(f)
        model_args = config.get('model_args', {})
        feeder_class = config.get('feeder', 'feeders.feeder_hockey.Feeder')
        test_feeder_args = config.get('test_feeder_args', {})
        if split == 'val' and config.get('val_feeder_args'):
            test_feeder_args = config['val_feeder_args']
    else:
        model_args = {
            'num_class': 11, 'num_point': 20, 'num_person': 1,
            'graph': 'graph.hockey.Graph',
            'graph_args': {'labeling_mode': 'spatial'},
            'head': ['ViT-B/32']
        }
        feeder_class = 'feeders.feeder_hockey.Feeder'
        test_feeder_args = {
            'data_path': './data/test_v2.pkl', 'split': 'test',
            'window_size': 64, 'normalization': True,
        }

    # Import and create model
    mod_str, _, cls_str = config.get('model', 'model.ctrgcn.Model_lst_4part_hockey').rpartition('.')
    __import__(mod_str)
    import sys
    Model = getattr(sys.modules[mod_str], cls_str)
    model = Model(**model_args).to(device)

    # Load weights
    ckpt_path = find_checkpoint(run_dir, epoch)
    print(f"Loading checkpoint: {ckpt_path}")
    weights = torch.load(ckpt_path, map_location=device)
    new_state = OrderedDict()
    for k, v in weights.items():
        name = k[7:] if k.startswith('module.') else k
        new_state[name] = v
    model.load_state_dict(new_state)
    model.eval()

    # Create data loader
    mod_str2, _, cls_str2 = feeder_class.rpartition('.')
    __import__(mod_str2)
    Feeder = getattr(sys.modules[mod_str2], cls_str2)
    dataset = Feeder(**test_feeder_args)
    loader = torch.utils.data.DataLoader(
        dataset, batch_size=64, shuffle=False, num_workers=4, drop_last=False
    )

    clip_head = model_args.get('head', ['ViT-B/32'])[0]
    return model, loader, clip_head


def encode_text_parts(clip_head, device):
    """Encode all 5 text sets (global + 4 parts) via CLIP text encoder."""
    _, num_text_aug, text_dict = text_prompt_hockey_pasta_pool_4part()

    clip_model, _ = clip.load(clip_head, device=device, jit=False)
    text_encoder = TextCLIP(clip_model).to(device)
    text_encoder.eval()

    # text_dict[aug_id] = tensor[num_classes, 77]
    text_embeddings = {}  # {aug_id: tensor[num_classes, embed_dim]}
    with torch.no_grad():
        for aug_id in range(num_text_aug):
            tokens = text_dict[aug_id].to(device)
            emb = text_encoder(tokens).float()
            emb = emb / emb.norm(dim=-1, keepdim=True)
            text_embeddings[aug_id] = emb

    return text_embeddings


def collect_features(model, loader, clip_head, device):
    """
    Run inference and collect per-sample features for each part.
    Returns: dict of {part_name: tensor[N, embed_dim]}, labels: tensor[N]
    """
    all_global = []
    all_head = []
    all_hand = []
    all_hip = []
    all_foot = []
    all_labels = []

    with torch.no_grad():
        for data, label, _ in tqdm(loader, desc='Collecting features', ncols=60):
            data = data.float().to(device)
            output, feature_dict, logit_scale, part_feature_list = model(data)

            global_feat = feature_dict[clip_head]  # [B, embed_dim]
            head_feat, hand_feat, hip_feat, foot_feat = part_feature_list

            all_global.append(global_feat.cpu())
            all_head.append(head_feat.cpu())
            all_hand.append(hand_feat.cpu())
            all_hip.append(hip_feat.cpu())
            all_foot.append(foot_feat.cpu())
            all_labels.append(label)

    features = {
        'global': torch.cat(all_global),
        'head': torch.cat(all_head),
        'hand': torch.cat(all_hand),
        'hip': torch.cat(all_hip),
        'foot': torch.cat(all_foot),
    }
    labels = torch.cat(all_labels)

    # Normalize
    for k in features:
        features[k] = features[k] / features[k].norm(dim=-1, keepdim=True)

    return features, labels


def compute_alignment_matrix(features, text_embeddings, labels):
    """
    Compute 5x5 alignment matrix: alignment[skel_part][text_part].
    For each sample, compute cosine sim between its skeleton part feature
    and the text embedding for its true class, averaged over all samples.
    """
    num_classes = text_embeddings[0].shape[0]
    skel_keys = ['global', 'head', 'hand', 'hip', 'foot']
    text_ids = [0, 1, 2, 3, 4]  # global, head, hand, hip, foot

    alignment = np.zeros((5, 5))

    for si, skel_name in enumerate(skel_keys):
        skel_feat = features[skel_name]  # [N, D]
        for ti, text_id in enumerate(text_ids):
            text_emb = text_embeddings[text_id].cpu()  # [C, D]
            # For each sample, get the text embedding of its true class
            sample_text = text_emb[labels.long()]  # [N, D]
            # Cosine similarity (both already normalized)
            cos_sim = (skel_feat * sample_text).sum(dim=-1)  # [N]
            alignment[si, ti] = cos_sim.mean().item()

    return alignment


def compute_per_class_alignment(features, text_embeddings, labels):
    """Compute per-class alignment for diagonal (matched) parts."""
    num_classes = text_embeddings[0].shape[0]
    skel_keys = ['global', 'head', 'hand', 'hip', 'foot']

    per_class = np.zeros((5, num_classes))

    for si, skel_name in enumerate(skel_keys):
        skel_feat = features[skel_name]
        text_emb = text_embeddings[si].cpu()  # matched text part

        for c in range(num_classes):
            mask = labels == c
            if mask.sum() == 0:
                continue
            class_skel = skel_feat[mask]  # [Nc, D]
            class_text = text_emb[c:c+1]  # [1, D]
            cos_sim = (class_skel * class_text).sum(dim=-1)
            per_class[si, c] = cos_sim.mean().item()

    return per_class


def main():
    parser = argparse.ArgumentParser(description='Verify text-to-joint alignment')
    parser.add_argument('--run_dir', required=True, help='Path to experiment work_dir')
    parser.add_argument('--epoch', type=int, default=None, help='Specific epoch (default: latest)')
    parser.add_argument('--split', default='val', help='Data split: val or test')
    parser.add_argument('--device', type=int, default=0)
    args = parser.parse_args()

    device = f"cuda:{args.device}" if torch.cuda.is_available() else "cpu"

    # Load model and data
    model, loader, clip_head = load_model_and_data(args.run_dir, args.epoch, args.split, device)

    # Encode text parts
    print("Encoding text parts via CLIP...")
    text_embeddings = encode_text_parts(clip_head, device)

    # Collect skeleton features
    features, labels = collect_features(model, loader, clip_head, device)
    print(f"Collected {labels.shape[0]} samples")

    # Compute 5x5 alignment matrix
    alignment = compute_alignment_matrix(features, text_embeddings, labels)

    # Print alignment matrix
    print("\n" + "=" * 75)
    print("Alignment Matrix (mean cosine similarity, matched by true class label)")
    print("=" * 75)
    col_header = f"{'Skel Part':<12s}" + "".join(f"{f'{n} text':>13s}" for n in PART_NAMES)
    print(col_header)
    print("-" * len(col_header))

    for si, skel_name in enumerate(PART_NAMES):
        row = f"{skel_name:<12s}"
        best_ti = np.argmax(alignment[si])
        for ti in range(5):
            val = alignment[si, ti]
            marker = " *" if ti == best_ti else "  "
            row += f"{val:10.4f}{marker} "
        print(row)

    # Diagnostic
    print("\nDiagnostic: Is each part most aligned with its own text?")
    all_correct = True
    for si, skel_name in enumerate(PART_NAMES):
        best_ti = np.argmax(alignment[si])
        best_name = PART_NAMES[best_ti]
        correct = best_ti == si
        if not correct:
            all_correct = False
        status = "YES" if correct else f"NO (best={best_name})"
        print(f"  {skel_name:>8s}: {status}")

    if all_correct:
        print("\n  All parts correctly aligned with their own text!")
    else:
        print("\n  WARNING: Some parts are NOT best-aligned with their own text.")
        print("  This suggests the contrastive alignment may not be working correctly.")

    # Per-class breakdown (diagonal only)
    per_class = compute_per_class_alignment(features, text_embeddings, labels)

    print("\n" + "=" * 75)
    print("Per-Class Alignment (diagonal: each part vs its own text)")
    print("=" * 75)
    col_header2 = f"{'Class':<30s}" + "".join(f"{n:>10s}" for n in PART_NAMES)
    print(col_header2)
    print("-" * len(col_header2))

    for c, cname in enumerate(CLASS_NAMES):
        row = f"{cname:<30s}"
        for si in range(5):
            row += f"{per_class[si, c]:10.4f}"
        print(row)

    # Mean
    print("-" * len(col_header2))
    row = f"{'MEAN':<30s}"
    for si in range(5):
        row += f"{per_class[si].mean():10.4f}"
    print(row)


if __name__ == '__main__':
    main()
