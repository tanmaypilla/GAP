"""
DEPRECATED: This script is no longer needed.

The combined_annotations_v2.pkl already contains pre-defined game-level
train/val/test splits. Use split_dataset_v2.py instead, which extracts
all three splits directly from the pkl.

Previously, this script took train_v2.pkl and randomly split it into
train_split_v2.pkl (85%) and val_split_v2.pkl (15%). This caused data
leakage because clips from the same game appeared in both train and val.
"""

import sys

def main():
    print("ERROR: This script is deprecated.")
    print("The val split is pre-defined in combined_annotations_v2.pkl (game-level).")
    print("Run split_dataset_v2.py instead to extract train/val/test splits.")
    sys.exit(1)


if __name__ == "__main__":
    main()
