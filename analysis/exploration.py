import pickle
import numpy as np
import matplotlib.pyplot as plt
import os

# CONFIG
FILE_PATH = "../../../../data/skating_actions_dataset/annotations/train.pkl"
OUTPUT_IMG = "skeleton_debug_v2.png"
INDEX_TO_PLOT = 0  # Use 0 since we know it exists and has data

def draw_skeleton_robust():
    if not os.path.exists(FILE_PATH):
        print(f"File not found: {FILE_PATH}")
        return

    with open(FILE_PATH, "rb") as f:
        data = pickle.load(f)

    # Get sample
    # Shape: (1, 30, 20, 2) -> Person 0, Frame 0
    sample = data['annotations'][INDEX_TO_PLOT]
    kps = sample['keypoint'][0][0] 
    
    # Filter out (0,0) empty points
    valid_kps = []
    valid_indices = []
    for i, (x, y) in enumerate(kps):
        if x > 1 and y > 1: # Simple filter for empty points
            valid_kps.append((x,y))
            valid_indices.append(i)
    
    if not valid_kps:
        print("Error: No valid keypoints found in this frame > (1,1).")
        return

    valid_kps = np.array(valid_kps)
    
    # Plotting
    plt.figure(figsize=(10, 10))
    
    # 1. Plot Points & Numbers
    plt.scatter(valid_kps[:, 0], valid_kps[:, 1], c='red', s=50)
    for i, (x, y) in zip(valid_indices, valid_kps):
        plt.text(x + 1, y + 1, str(i), fontsize=14, color='blue', weight='bold')

    # 2. Auto-scale margins so points aren't on the edge
    plt.margins(0.2)
    
    # 3. Invert Y axis (Image coordinates: 0 is top)
    plt.gca().invert_yaxis()
    
    plt.title(f"Skeleton Topology (Indices 17, 18, 19?)")
    plt.xlabel("X Pixel (Original Frame)")
    plt.ylabel("Y Pixel (Original Frame)")
    plt.grid(True)
    
    # Save
    plt.savefig(OUTPUT_IMG)
    print(f"\n[SUCCESS] Debug image saved to: {os.path.abspath(OUTPUT_IMG)}")
    print("Please open this image and locate points 17, 18, and 19.")

if __name__ == "__main__":
    draw_skeleton_robust()