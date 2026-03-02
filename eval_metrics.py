#!/usr/bin/env python
"""
Evaluate a hockey run: total accuracy, mean class accuracy, per-class accuracy.

Usage:
    python eval_metrics.py --run_dir work_dir/hockey/baseline_joint
    python eval_metrics.py --run_dir work_dir/hockey/baseline_joint --epoch 65
"""

import argparse
import os
import re
import glob
import yaml
import numpy as np
import torch
from collections import OrderedDict
from sklearn.metrics import confusion_matrix
from tqdm import tqdm

from model.ctrgcn import Model_lst_4part_hockey as Model
from feeders.feeder_hockey import Feeder


LABEL_NAMES = [
    'GLID_FORW', 'ACCEL_FORW', 'GLID_BACK', 'ACCEL_BACK',
    'TRANS_FORW_TO_BACK', 'TRANS_BACK_TO_FORW', 'POST_WHISTLE_GLIDING',
    'FACEOFF_BODY_POSITION', 'MAINTAIN_POSITION', 'PRONE', 'ON_A_KNEE',
]


def parse_epoch(filename):
    match = re.search(r'runs-(\d+)-', os.path.basename(filename))
    return int(match.group(1)) if match else -1


def find_checkpoint(run_dir, epoch=None):
    pt_files = glob.glob(os.path.join(run_dir, 'runs-*.pt'))
    if not pt_files:
        raise FileNotFoundError(f"No checkpoints found in {run_dir}")
    if epoch is not None:
        matches = [f for f in pt_files if parse_epoch(f) == epoch]
        if not matches:
            raise FileNotFoundError(f"No checkpoint for epoch {epoch} in {run_dir}")
        return matches[0]
    # Default: highest epoch
    return max(pt_files, key=parse_epoch)


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument('--run_dir', required=True, help='Path to work_dir run folder')
    parser.add_argument('--epoch', type=int, default=None, help='Epoch to evaluate (default: latest)')
    parser.add_argument('--device', type=int, default=0)
    args = parser.parse_args()

    # Load config
    config_path = os.path.join(args.run_dir, 'config.yaml')
    with open(config_path, 'r') as f:
        config = yaml.safe_load(f)

    # Find checkpoint
    ckpt_path = find_checkpoint(args.run_dir, args.epoch)
    ckpt_epoch = parse_epoch(ckpt_path)
    print(f"Run:        {args.run_dir}")
    print(f"Checkpoint: epoch {ckpt_epoch} ({os.path.basename(ckpt_path)})")

    # Setup model
    dev = f"cuda:{args.device}" if torch.cuda.is_available() else "cpu"
    model_args = config['model_args']
    model = Model(**model_args).to(dev)

    weights = torch.load(ckpt_path, map_location=dev)
    new_state = OrderedDict()
    for k, v in weights.items():
        name = k[7:] if k.startswith('module.') else k
        new_state[name] = v
    model.load_state_dict(new_state)
    model.eval()

    # Setup data loader
    test_args = config['test_feeder_args']
    dp = test_args.get('data_path', '')
    if dp and not os.path.isabs(dp) and not os.path.isfile(dp):
        # Relative path doesn't resolve from CWD; try resolving from run_dir
        alt = os.path.normpath(os.path.join(args.run_dir, dp))
        if os.path.isfile(alt):
            test_args['data_path'] = alt
    dataset = Feeder(**test_args)
    loader = torch.utils.data.DataLoader(
        dataset, batch_size=config.get('test_batch_size', 64),
        shuffle=False, num_workers=4, drop_last=False
    )

    # Inference
    # Some models were trained before the feeder's /1000 scaling fix and have BN
    # running stats at the original scale (~tens). Others were trained after the fix
    # with stats at ~0.01-0.1 scale. Auto-detect by checking BN running_mean magnitude.
    bn_mean_mag = model.data_bn.running_mean.abs().max().item()
    rescale = test_args.get('normalization', False) and bn_mean_mag > 1.0
    if rescale:
        print("Note: undoing /1000 scaling (model trained at original scale)")

    all_preds = []
    all_labels = []
    with torch.no_grad():
        for data, label, _ in tqdm(loader, desc="Evaluating", ncols=60):
            data = data.float().to(dev)
            if rescale:
                data = data * 1000.0
            output, _, _, _ = model(data)
            _, pred = torch.max(output, 1)
            all_preds.append(pred.cpu().numpy())
            all_labels.append(label.cpu().numpy())

    all_preds = np.concatenate(all_preds)
    all_labels = np.concatenate(all_labels)
    num_classes = len(LABEL_NAMES)

    # Metrics
    cm = confusion_matrix(all_labels, all_preds, labels=range(num_classes))
    per_class_correct = cm.diagonal()
    per_class_total = cm.sum(axis=1)

    with np.errstate(divide='ignore', invalid='ignore'):
        per_class_acc = np.where(per_class_total > 0,
                                 per_class_correct / per_class_total, 0.0)

    mask = all_labels < num_classes
    total_acc = (all_preds[mask] == all_labels[mask]).sum() / mask.sum() * 100
    mean_class_acc = per_class_acc.mean() * 100

    # Print table
    print()
    print(f"{'Class':<28} {'Samples':>8} {'Correct':>8} {'Accuracy':>10}")
    print("-" * 58)
    for i, name in enumerate(LABEL_NAMES):
        acc = per_class_acc[i] * 100
        print(f"{name:<28} {per_class_total[i]:>8d} {per_class_correct[i]:>8d} {acc:>9.2f}%")
    print("-" * 58)
    print(f"{'Mean class accuracy':<28} {'':>8} {'':>8} {mean_class_acc:>9.2f}%")
    print(f"{'Total accuracy':<28} {mask.sum():>8d} {(all_preds[mask] == all_labels[mask]).sum():>8d} {total_acc:>9.2f}%")


if __name__ == '__main__':
    main()
