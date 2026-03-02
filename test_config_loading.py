import yaml
import argparse
import os
import sys

# Import the parser from your main script to ensure consistency
from main_multipart_hockey import get_parser

def test_config():
    print("=== Phase 4 Test: Configuration Loading ===")
    
    config_path = "config/hockey/default.yaml"
    if not os.path.exists(config_path):
        print(f"[FAIL] Config file not found at {config_path}")
        return

    # 1. Load YAML directly
    with open(config_path, 'r') as f:
        yaml_args = yaml.load(f, Loader=yaml.SafeLoader)
    print(f"[PASS] Valid YAML syntax in {config_path}")

    # 2. Simulate Argument Parsing (what main.py does)
    parser = get_parser()
    
    # Mock command line args: run the script pointing to the config
    sys.argv = ['main_multipart_hockey.py', '--config', config_path]
    
    try:
        p = parser.parse_args()
        
        # Inject yaml args into parser namespace (mimicking main.py logic)
        parser.set_defaults(**yaml_args)
        args = parser.parse_args()
        
        print("[PASS] Argument Parser successfully merged Config file.")
        
        # 3. Critical Value Checks
        checks = [
            ('num_point', args.model_args['num_point'], 20),
            ('num_class', args.model_args['num_class'], 12),
            ('feeder', args.feeder, 'feeders.feeder_hockey.Feeder'),
            ('normalization', args.train_feeder_args['normalization'], True)
        ]
        
        print("\nVerifying Critical Parameters:")
        all_ok = True
        for name, val, expected in checks:
            if val == expected:
                print(f"  > {name}: {val} [OK]")
            else:
                print(f"  > {name}: {val} [FAIL] (Expected {expected})")
                all_ok = False
        
        if all_ok:
            print("\n[SUCCESS] Configuration is valid and ready for training.")
        
    except Exception as e:
        print(f"[FAIL] Configuration parsing error: {e}")

if __name__ == "__main__":
    test_config()