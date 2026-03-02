import re
import matplotlib.pyplot as plt
import argparse
import os

def parse_log(file_path):
    epochs = []
    train_loss = []
    train_acc = []
    test_loss = []
    test_top1 = []
    test_top5 = []
    val_mean_class_acc = []

    current_epoch = 0

    with open(file_path, 'r') as f:
        lines = f.readlines()

    for line in lines:
        # Detect Epoch
        epoch_match = re.search(r'Training epoch:\s+(\d+)', line)
        if epoch_match:
            current_epoch = int(epoch_match.group(1))
            if current_epoch not in epochs:
                epochs.append(current_epoch)

        # Detect Training Metrics
        # Example: Mean training loss: 0.5947.  Mean training acc: 80.45%.
        train_match = re.search(r'Mean training loss:\s+([\d\.]+)\.\s+Mean training acc:\s+([\d\.]+)%', line)
        if train_match:
            train_loss.append(float(train_match.group(1)))
            train_acc.append(float(train_match.group(2)))

        # Detect Test Loss
        # Example: Mean test loss of 86 batches: 0.7180293788050496.
        test_loss_match = re.search(r'Mean (?:test|val) loss of \d+ batches:\s+([\d\.]+).', line)
        if test_loss_match:
            test_loss.append(float(test_loss_match.group(1)))

        # Detect Top-1
        # Example: Top1: 77.66%
        top1_match = re.search(r'Top1:\s+([\d\.]+)%', line)
        if top1_match:
            test_top1.append(float(top1_match.group(1)))

        # Detect Top-5
        # Example: Top5: 98.46%
        top5_match = re.search(r'Top[35]:\s+([\d\.]+)%', line)
        if top5_match:
            test_top5.append(float(top5_match.group(1)))

        # Detect Mean Class Accuracy
        # Example: Mean class accuracy: 43.94%
        mca_match = re.search(r'Mean class accuracy:\s+([\d\.]+)%', line)
        if mca_match:
            val_mean_class_acc.append(float(mca_match.group(1)))

    # Align lengths (handle partial epochs at the end)
    min_len = min(len(train_loss), len(test_loss), len(test_top1), len(val_mean_class_acc))
    return {
        'epochs': epochs[:min_len],
        'train_loss': train_loss[:min_len],
        'train_acc': train_acc[:min_len],
        'test_loss': test_loss[:min_len],
        'test_top1': test_top1[:min_len],
        'test_top5': test_top5[:min_len],
        'val_mean_class_acc': val_mean_class_acc[:min_len]
    }

def plot_metrics(data, output_dir):
    epochs = data['epochs']
    
    # Setup styles
    plt.style.use('seaborn-v0_8-darkgrid')
    
    # 1. Loss Curve
    plt.figure(figsize=(10, 6))
    plt.plot(epochs, data['train_loss'], label='Training Loss', color='tab:blue', linewidth=2)
    plt.plot(epochs, data['test_loss'], label='Val Loss', color='tab:orange', linewidth=2, linestyle='--')
    plt.title('Loss Curve: Train vs Val', fontsize=14)
    plt.xlabel('Epoch')
    plt.ylabel('Loss')
    plt.legend()
    plt.savefig(os.path.join(output_dir, 'loss_curve.png'))
    print(f"Saved {output_dir}/loss_curve.png")
    plt.close()

    # 2. Accuracy Curve (Train vs Test Top-1)
    plt.figure(figsize=(10, 6))
    plt.plot(epochs, data['train_acc'], label='Training Acc', color='tab:green', linewidth=2)
    plt.plot(epochs, data['test_top1'], label='Val Top-1 Acc', color='tab:red', linewidth=2)
    plt.plot(epochs, data['val_mean_class_acc'], label='Val Mean-Class Acc', color='tab:orange', linewidth=2, linestyle='--')
    plt.title('Accuracy: Training vs Val Top-1', fontsize=14)
    plt.xlabel('Epoch')
    plt.ylabel('Accuracy (%)')
    plt.legend()
    plt.savefig(os.path.join(output_dir, 'acc_curve.png'))
    print(f"Saved {output_dir}/acc_curve.png")
    plt.close()

    # 3. Top-K Gap (Top-1 vs Top-5)
    plt.figure(figsize=(10, 6))
    plt.plot(epochs, data['test_top1'], label='Val Top-1', color='tab:red', linewidth=2)
    plt.plot(epochs, data['test_top5'], label='Val Top-3/5', color='tab:purple', linewidth=2, linestyle='-.')
    plt.fill_between(epochs, data['test_top1'], data['test_top5'], color='gray', alpha=0.1, label='The GAP')
    plt.title('Top-1 vs Top-K Accuracy', fontsize=14)
    plt.xlabel('Epoch')
    plt.ylabel('Accuracy (%)')
    plt.legend()
    plt.savefig(os.path.join(output_dir, 'topk_gap.png'))
    print(f"Saved {output_dir}/topk_gap.png")
    plt.close()

if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument('--log', type=str, required=True, help='Path to the log.txt file')
    parser.add_argument('--out', type=str, default='.', help='Directory to save plots')
    args = parser.parse_args()

    output_dir = os.path.join(args.out, 'analysis')
    if not os.path.exists(output_dir):
        os.makedirs(output_dir)

    data = parse_log(args.log)
    plot_metrics(data, output_dir)

# python plot_logs.py --log work_dir/hockey/baseline_joint/log.txt --out work_dir/hockey/baseline_joint/
# python plot_logs.py --log work_dir/hockey/text_guided_0.8/log.txt --out work_dir/hockey/text_guided_0.8/