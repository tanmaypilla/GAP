import sys
import os
import numpy as np

# 1. Setup Path so Python can find 'feeders' module
sys.path.append(os.getcwd()) 

try:
    from feeders.feeder_hockey import Feeder
except ImportError:
    print("Error: Could not import Feeder. Make sure you are running this from the 'GAP/' folder.")
    sys.exit(1)

# CONFIG
DATA_PATH = "/data/skating_actions_dataset/annotations/train.pkl"

def test_modality(name, **kwargs):
    print(f"\n=== Testing Modality: {name} ===")
    try:
        dataset = Feeder(data_path=DATA_PATH, window_size=64, **kwargs)
        print(f"  > Dataset Length: {len(dataset)}")
        
        # Get first sample
        data, label, index = dataset[0]
        
        # Checks
        shape_ok = data.shape == (2, 64, 20, 1)
        value_range = np.max(data) - np.min(data)
        has_zeros = np.all(data == 0)
        
        print(f"  > Shape: {data.shape} [{'OK' if shape_ok else 'FAIL'}]")
        print(f"  > Range: Min {np.min(data):.2f} / Max {np.max(data):.2f}")
        print(f"  > Label: {label}")
        
        if has_zeros:
            print("  > WARNING: Data is all zeros! Check your calculation logic.")
        else:
            print("  > Data looks valid (non-zero).")
            
    except Exception as e:
        print(f"  > CRITICAL FAILURE: {e}")
        import traceback
        traceback.print_exc()

if __name__ == "__main__":
    if not os.path.exists(DATA_PATH):
        print(f"Error: Data file not found at {DATA_PATH}")
        sys.exit(1)

    # 1. Test Standard Joint (Normalization ON)
    test_modality("Joints (Normalized)", normalization=True, bone=False, vel=False)

    # 2. Test Bone (Should calculate differences between joints)
    test_modality("Bone", normalization=True, bone=True, vel=False)

    # 3. Test Velocity (Should calculate differences between frames)
    test_modality("Velocity", normalization=True, bone=False, vel=True)
    
    print("\n[SUCCESS] Feeder test complete.")