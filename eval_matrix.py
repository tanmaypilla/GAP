import argparse
import torch
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.metrics import confusion_matrix
import sys
import os
import glob
import re
import yaml
from tqdm import tqdm

# Import your model structure
from model.ctrgcn import Model_lst_4part_hockey as Model
from feeders.feeder_hockey import Feeder

def get_parser():
    parser = argparse.ArgumentParser()
    # Path to the folder containing your .pt files
    parser.add_argument('--run_dir', default='./work_dir/hockey/baseline_joint/')
    parser.add_argument('--device', type=int, default=0)
    return parser

def parse_epoch(filename):
    # Extracts epoch number from 'runs-48-15408.pt'
    match = re.search(r'runs-(\d+)-', filename)
    if match:
        return int(match.group(1))
    return -1

def load_label_names(path='./text/hockey_label_map.txt'):
    try:
        with open(path, 'r') as f:
            labels = [line.strip() for line in f.readlines()]
        return labels
    except:
        return [str(i) for i in range(12)]

def run_analysis(args):
    dev = f"cuda:{args.device}"
    label_names = load_label_names()
    num_class = len(label_names)
    
    # 1. Locate Checkpoints
    runs_path = os.path.join(args.run_dir, '')
    pt_files = glob.glob(os.path.join(runs_path, '*.pt'))
    
    # Sort by epoch
    pt_files.sort(key=parse_epoch)
    
    print(f"[INFO] Found {len(pt_files)} checkpoints to analyze.")

    # 2. Setup Data Loader (Once)
    dataset = Feeder(
        data_path='/data/skating_actions_dataset/annotations/test.pkl',
        split='test',
        window_size=64,
        normalization=True,
        debug=False
    )
    loader = torch.utils.data.DataLoader(
        dataset, batch_size=64, shuffle=False, num_workers=4, drop_last=False
    )

    # 3. Setup Model (Once)
    # Load config to get args
    config_path = os.path.join(args.run_dir, 'config.yaml')
    if os.path.exists(config_path):
        with open(config_path, 'r') as f:
            config = yaml.safe_load(f)
        model_args = config.get('model_args', {})
    else:
        model_args = {'num_class': num_class, 'num_point': 20, 'num_person': 1, 
                      'graph': 'graph.hockey.Graph', 'graph_args': {'labeling_mode': 'spatial'},
                      'head': ['ViT-B/32']}

    model = Model(**model_args).to(dev)
    
    # Storage for history
    history = {name: [] for name in label_names}
    epochs = []

    # 4. Loop Through Epochs
    for pt_file in tqdm(pt_files, desc="Analyzing Epochs"):
        epoch = parse_epoch(os.path.basename(pt_file))
        epochs.append(epoch)
        
        # Load Weights
        weights = torch.load(pt_file)
        from collections import OrderedDict
        new_state_dict = OrderedDict()
        for k, v in weights.items():
            name = k[7:] if k.startswith('module.') else k 
            new_state_dict[name] = v
        model.load_state_dict(new_state_dict)
        model.eval()
        
        # Inference
        all_preds = []
        all_labels = []
        with torch.no_grad():
            for data, label, _ in loader:
                data = data.float().to(dev)
                output, _, _, _ = model(data)
                _, pred = torch.max(output, 1)
                all_preds.append(pred.cpu().numpy())
                all_labels.append(label.cpu().numpy())
        
        # Metrics
        all_preds = np.concatenate(all_preds)
        all_labels = np.concatenate(all_labels)
        cm = confusion_matrix(all_labels, all_preds, labels=range(num_class))
        
        # Per-Class Acc (Diagonal / Row Sum)
        with np.errstate(divide='ignore', invalid='ignore'):
            per_class_acc = cm.diagonal() / cm.sum(axis=1)
            per_class_acc = np.nan_to_num(per_class_acc)
            
        for i, name in enumerate(label_names):
            history[name].append(per_class_acc[i] * 100)

    # 5. Plotting
    plt.figure(figsize=(14, 8))
    
    # We split into 'Major' and 'Minor' classes for clarity if needed, 
    # but for now, let's plot all.
    for name in label_names:
        # Highlight the dominant class
        if name == 'GLID_FORW':
            plt.plot(epochs, history[name], label=name, linewidth=3, linestyle='--')
        else:
            plt.plot(epochs, history[name], label=name, alpha=0.7)
            
    plt.xlabel('Epoch')
    plt.ylabel('Accuracy (%)')
    plt.title('Per-Class Accuracy History (Did the minor classes ever learn?)')
    plt.legend(bbox_to_anchor=(1.05, 1), loc='upper left')
    plt.grid(True, alpha=0.3)
    plt.tight_layout()
    plt.savefig('class_history.png')
    print("[SUCCESS] Saved plot to class_history.png")

if __name__ == "__main__":
    parser = get_parser()
    args = parser.parse_args()
    run_analysis(args)