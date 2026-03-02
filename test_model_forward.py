import torch
from model.ctrgcn import Model_lst_4part_hockey

def test_model_forward():
    print("=== Phase 3 Test: Model Architecture Forward Pass (CPU Mode) ===")
    
    # 1. Configuration matching 'config/hockey/default.yaml'
    args = {
        'num_class': 12,
        'num_point': 20,
        'num_person': 1,
        'graph': 'graph.hockey.Graph',
        'graph_args': {'labeling_mode': 'spatial'},
        'in_channels': 2,
        'head': ['ViT-B/32'] 
    }

    # 2. Initialize Model (On CPU)
    try:
        model = Model_lst_4part_hockey(**args)
        print("[PASS] Model initialized successfully.")
        
        # Calculate parameters
        params = sum(p.numel() for p in model.parameters())
        print(f"  > Parameters: {params/1e6:.2f}M")
    except Exception as e:
        print(f"[FAIL] Model initialization failed: {e}")
        import traceback
        traceback.print_exc()
        return

    # 3. Create Dummy Data
    # Shape: (Batch, Channels, Frames, Vertices, Persons)
    # Hockey: (2, 2, 64, 20, 1)
    dummy_input = torch.randn(2, 2, 64, 20, 1)
    
    # 4. Forward Pass
    print("\nRunning Forward Pass...")
    try:
        # Output: (logits, feature_dict, logit_scale, part_features)
        logits, feats, scale, parts = model(dummy_input)
        
        print("[PASS] Forward pass successful.")
        print(f"  > Logits Shape: {logits.shape} (Expected: [2, 12])")
        print(f"  > Part Features: {len(parts)} parts returned (Head, Hand, Hip, Foot).")
        
        if logits.shape == (2, 12):
            print("  > Dimensions match exactly.")
        else:
            print(f"  > WARNING: Dimension mismatch. Expected [2, 12], got {logits.shape}")
            
    except Exception as e:
        print(f"[FAIL] Forward pass error: {e}")
        import traceback
        traceback.print_exc()

if __name__ == "__main__":
    test_model_forward()