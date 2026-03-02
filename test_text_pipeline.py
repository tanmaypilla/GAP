import clip
import torch
import numpy as np
from Text_Prompt import text_prompt_hockey

def test_text_pipeline():
    print("=== Phase 2 Test: Text Pipeline & CLIP Tokenization ===")
    
    # 1. Load Prompts
    try:
        prompts = text_prompt_hockey()
        print(f"[PASS] Loaded prompts for {len(prompts)} classes.")
    except Exception as e:
        print(f"[FAIL] Could not load text prompts: {e}")
        return

    # 2. Inspect Content
    print(f"  > Class 0 Prompts: {len(prompts[0])} variations found.")
    print(f"  > Sample: '{prompts[0][0]}'")

    # 3. Test Tokenization
    print("\nTesting CLIP Tokenization...")
    device = "cpu" # Keep it simple for testing
    
    try:
        # Flatten list to test all possible prompts
        all_sentences = [p for class_list in prompts for p in class_list]
        
        # Tokenize (CLIP max context is 77 tokens)
        # We use truncate=True to ensure long PASTA descriptions don't crash it
        tokens = clip.tokenize(all_sentences, truncate=True).to(device)
        
        print(f"[PASS] Successfully tokenized {len(all_sentences)} text prompts.")
        print(f"  > Token Shape: {tokens.shape} (N, 77)")
        
    except Exception as e:
        print(f"[FAIL] Tokenization error: {e}")
        print("  Hint: Check if any description in your .txt files is extremely long/malformed.")

if __name__ == "__main__":
    test_text_pipeline()
    