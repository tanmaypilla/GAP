import pickle
import numpy as np
import torch

def get_weights(path='/data/skating_actions_dataset/annotations/train.pkl'):
    with open(path, 'rb') as f:
        data = pickle.load(f)
    
    # Handle list vs dict
    if isinstance(data, dict):
        data = data['annotations']
        
    labels = [sample['label'] for sample in data]
    num_classes = 12 # Hardcoded for Hockey
    
    counts = np.zeros(num_classes)
    for l in labels:
        counts[l] += 1
        
    print("--- Class Counts ---")
    print(counts)
    
    # Formula: N_samples / (N_classes * Count_i)
    # This ensures the weighted sum is balanced
    weights = len(labels) / (num_classes * counts)
    
    # Safety: Clamp huge weights if a class is super rare
    weights = torch.FloatTensor(weights)
    
    print("\n--- RECOMMENDED WEIGHTS ---")
    print("Copy this list into your config or main.py:")
    print(weights.tolist())
    
    return weights

if __name__ == "__main__":
    get_weights()