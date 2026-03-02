"""
Extract train/val/test splits from combined_annotations_v2.pkl using the
pre-defined game-level splits stored in data['split'].

The v2 dataset has 11 classes (labels 0-10) with RAPID_DECELERATION removed.

Split breakdown (from the pkl):
  - Train: 12 games → ~20,390 samples
  - Test:  3 games  → ~5,383 samples
  - Val:   1 game   → ~1,864 samples
  - Zero game overlap between splits

Output format matches the feeder expectation: {'annotations': [list of sample dicts]}
"""

import os
import pickle
import re
from collections import Counter

ANNOTATIONS_DIR = "/data/skating_actions_dataset/annotations"
OUTPUT_DIR = "/home/tanmay-ura/GAP/data"
INPUT_FILE = f"{ANNOTATIONS_DIR}/combined_annotations_v2.pkl"

TRAIN_FILE = f"{OUTPUT_DIR}/train_split_v2.pkl"
VAL_FILE = f"{OUTPUT_DIR}/val_split_v2.pkl"
TEST_FILE = f"{OUTPUT_DIR}/test_v2.pkl"

CLASS_NAMES = {
    0: "GLID_FORW",
    1: "ACCEL_FORW",
    2: "GLID_BACK",
    3: "ACCEL_BACK",
    4: "TRANS_FORW_TO_BACK",
    5: "TRANS_BACK_TO_FORW",
    6: "POST_WHISTLE_GLIDING",
    7: "FACEOFF_BODY_POSITION",
    8: "MAINTAIN_POSITION",
    9: "PRONE",
    10: "ON_A_KNEE",
}


def get_game(sample_name):
    """Extract game identifier from sample name."""
    match = re.match(r'(.+?)_\d+_\d+_\d+$', sample_name)
    return match.group(1) if match else sample_name


def main():
    with open(INPUT_FILE, "rb") as f:
        data = pickle.load(f)

    annotations = data["annotations"]
    split = data["split"]

    print(f"Loaded {len(annotations)} total samples from {INPUT_FILE}")
    print(f"Pre-defined splits: train={len(split['train'])}, "
          f"test={len(split['test'])}, val={len(split['val'])}")
    print()

    os.makedirs(OUTPUT_DIR, exist_ok=True)

    # Build lookup: frame_dir (sample name) -> annotation dict
    name_to_ann = {a["frame_dir"]: a for a in annotations}

    # Extract annotations for each split
    split_data = {}
    for split_name, sample_ids in split.items():
        anns = [name_to_ann[sid] for sid in sample_ids if sid in name_to_ann]
        missing = len(sample_ids) - len(anns)
        if missing > 0:
            print(f"WARNING: {missing} sample IDs in '{split_name}' not found in annotations")
        split_data[split_name] = anns

    # Save in feeder-compatible format
    for split_name, out_path in [("train", TRAIN_FILE), ("val", VAL_FILE), ("test", TEST_FILE)]:
        anns = split_data[split_name]
        with open(out_path, "wb") as f:
            pickle.dump({"annotations": anns}, f)
        print(f"Saved {len(anns):>6d} {split_name:<5s} samples to {out_path}")

    # Print per-class counts and games per split
    for split_name in ["train", "val", "test"]:
        anns = split_data[split_name]
        labels = [a["label"] for a in anns]
        counts = Counter(labels)
        games = sorted(set(get_game(a["frame_dir"]) for a in anns))

        print(f"\n{split_name.upper()} split ({len(anns)} samples, {len(games)} games):")
        print(f"  Games: {', '.join(games)}")
        for cls_id in sorted(counts.keys()):
            name = CLASS_NAMES.get(cls_id, f"UNKNOWN_{cls_id}")
            print(f"  {cls_id:2d}: {name:<28s} {counts[cls_id]:>6d}")


if __name__ == "__main__":
    main()
