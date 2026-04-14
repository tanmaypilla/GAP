#!/usr/bin/env python
"""
Compare two experiments (baseline vs text-guided) using their saved confusion matrix CSVs.

Usage:
    python compare_experiments.py \
        --baseline_dir work_dir/hockey/baseline_v2 \
        --textguided_dir work_dir/hockey/gap_v2 \
        [--epoch N] [--split val]
"""

import argparse
import csv
import glob
import os
import re

import matplotlib.pyplot as plt
import numpy as np
import seaborn as sns


def find_best_epoch_csv(run_dir, split='val'):
    """Find the CSV with the highest epoch number for the given split."""
    pattern = os.path.join(run_dir, f'epoch*_{split}_each_class_acc.csv')
    csvs = glob.glob(pattern)
    if not csvs:
        raise FileNotFoundError(f"No CSVs matching {pattern}")
    def extract_epoch(path):
        m = re.search(r'epoch(\d+)_', os.path.basename(path))
        return int(m.group(1)) if m else -1
    csvs.sort(key=extract_epoch)
    return csvs[-1], extract_epoch(csvs[-1])


def load_csv(path):
    """
    Load a per-class accuracy CSV.
    Row 0: class name headers
    Row 1: per-class accuracy floats
    Rows 2+: confusion matrix (integer counts)
    Returns: class_names, per_class_acc, confusion_matrix
    """
    with open(path, 'r') as f:
        reader = list(csv.reader(f))
    class_names = reader[0]
    per_class_acc = np.array([float(x) for x in reader[1]])
    cm = np.array([[int(x) for x in row] for row in reader[2:]])
    return class_names, per_class_acc, cm


def compute_metrics(cm):
    """Compute per-class precision, recall, F1, and support from confusion matrix."""
    n = cm.shape[0]
    precision = np.zeros(n)
    recall = np.zeros(n)
    f1 = np.zeros(n)
    support = cm.sum(axis=1)

    for i in range(n):
        tp = cm[i, i]
        fp = cm[:, i].sum() - tp
        fn = cm[i, :].sum() - tp
        precision[i] = tp / max(tp + fp, 1)
        recall[i] = tp / max(tp + fn, 1)
        if precision[i] + recall[i] > 0:
            f1[i] = 2 * precision[i] * recall[i] / (precision[i] + recall[i])
    return precision, recall, f1, support


def print_comparison(class_names, prec_b, rec_b, f1_b, prec_t, rec_t, f1_t):
    """Print a formatted comparison table."""
    header = (f"{'Class':<30s} {'Prec_B':>7s} {'Prec_T':>7s} {'Rec_B':>7s} "
              f"{'Rec_T':>7s} {'F1_B':>7s} {'F1_T':>7s} {'Delta':>7s} {'Verdict':>8s}")
    print(header)
    print('-' * len(header))

    for i, name in enumerate(class_names):
        delta = f1_t[i] - f1_b[i]
        if abs(delta) < 0.02:
            verdict = 'SAME'
        elif delta > 0:
            verdict = 'BETTER'
        else:
            verdict = 'WORSE'
        print(f"{name:<30s} {prec_b[i]:6.1%} {prec_t[i]:6.1%} "
              f"{rec_b[i]:6.1%} {rec_t[i]:6.1%} "
              f"{f1_b[i]:6.1%} {f1_t[i]:6.1%} "
              f"{delta:+6.1%} {verdict:>8s}")

    # Macro averages
    print('-' * len(header))
    delta_macro = f1_t.mean() - f1_b.mean()
    print(f"{'MACRO AVERAGE':<30s} {prec_b.mean():6.1%} {prec_t.mean():6.1%} "
          f"{rec_b.mean():6.1%} {rec_t.mean():6.1%} "
          f"{f1_b.mean():6.1%} {f1_t.mean():6.1%} "
          f"{delta_macro:+6.1%}")


def plot_confusion_matrices(class_names, cm_b, cm_t, epoch_b, epoch_t, out_path):
    """Plot side-by-side normalized CMs and a delta heatmap."""
    # Normalize by row (true label)
    with np.errstate(divide='ignore', invalid='ignore'):
        cm_b_norm = cm_b.astype(float) / np.maximum(cm_b.sum(axis=1, keepdims=True), 1)
        cm_t_norm = cm_t.astype(float) / np.maximum(cm_t.sum(axis=1, keepdims=True), 1)

    short_names = [n[:12] for n in class_names]

    fig, axes = plt.subplots(1, 3, figsize=(24, 7))

    # Baseline
    sns.heatmap(cm_b_norm, annot=True, fmt='.2f', cmap='Blues',
                xticklabels=short_names, yticklabels=short_names,
                ax=axes[0], vmin=0, vmax=1, cbar_kws={'shrink': 0.8})
    axes[0].set_title(f'Baseline (epoch {epoch_b})')
    axes[0].set_ylabel('True')
    axes[0].set_xlabel('Predicted')

    # Text-guided
    sns.heatmap(cm_t_norm, annot=True, fmt='.2f', cmap='Blues',
                xticklabels=short_names, yticklabels=short_names,
                ax=axes[1], vmin=0, vmax=1, cbar_kws={'shrink': 0.8})
    axes[1].set_title(f'Text-Guided (epoch {epoch_t})')
    axes[1].set_ylabel('True')
    axes[1].set_xlabel('Predicted')

    # Delta
    delta = cm_t_norm - cm_b_norm
    vmax = max(abs(delta.min()), abs(delta.max()), 0.1)
    sns.heatmap(delta, annot=True, fmt='+.2f', cmap='RdBu_r',
                xticklabels=short_names, yticklabels=short_names,
                ax=axes[2], vmin=-vmax, vmax=vmax, cbar_kws={'shrink': 0.8})
    axes[2].set_title('Delta (Text - Baseline)')
    axes[2].set_ylabel('True')
    axes[2].set_xlabel('Predicted')

    plt.tight_layout()
    plt.savefig(out_path, dpi=150, bbox_inches='tight')
    print(f"\nSaved comparison plot to: {out_path}")
    plt.close()


def main():
    parser = argparse.ArgumentParser(description='Compare baseline vs text-guided experiments')
    parser.add_argument('--baseline_dir', required=True, help='Path to baseline work_dir')
    parser.add_argument('--textguided_dir', required=True, help='Path to text-guided work_dir')
    parser.add_argument('--epoch', type=int, default=None, help='Specific epoch to compare (default: best/last)')
    parser.add_argument('--split', default='val', help='Eval split: val or test')
    parser.add_argument('--output', default=None, help='Output plot path (default: auto)')
    args = parser.parse_args()

    # Load CSVs
    if args.epoch is not None:
        csv_b = os.path.join(args.baseline_dir, f'epoch{args.epoch}_{args.split}_each_class_acc.csv')
        csv_t = os.path.join(args.textguided_dir, f'epoch{args.epoch}_{args.split}_each_class_acc.csv')
        epoch_b = epoch_t = args.epoch
    else:
        csv_b, epoch_b = find_best_epoch_csv(args.baseline_dir, args.split)
        csv_t, epoch_t = find_best_epoch_csv(args.textguided_dir, args.split)

    print(f"Baseline:    {csv_b} (epoch {epoch_b})")
    print(f"Text-guided: {csv_t} (epoch {epoch_t})")
    print()

    class_names_b, acc_b, cm_b = load_csv(csv_b)
    class_names_t, acc_t, cm_t = load_csv(csv_t)

    assert class_names_b == class_names_t, "Class names mismatch between experiments"
    class_names = class_names_b

    # Compute metrics
    prec_b, rec_b, f1_b, support_b = compute_metrics(cm_b)
    prec_t, rec_t, f1_t, support_t = compute_metrics(cm_t)

    # Print comparison
    print_comparison(class_names, prec_b, rec_b, f1_b, prec_t, rec_t, f1_t)

    # Top-1 accuracy summary
    print(f"\nOverall Top-1:  Baseline={acc_b.mean():.2%}  Text-Guided={acc_t.mean():.2%}  "
          f"Delta={acc_t.mean() - acc_b.mean():+.2%}")

    # Plot
    out_path = args.output or os.path.join(args.textguided_dir, 'comparison_vs_baseline.png')
    plot_confusion_matrices(class_names, cm_b, cm_t, epoch_b, epoch_t, out_path)


if __name__ == '__main__':
    main()
