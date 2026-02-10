"""
Phase 1: Dataset structure analysis for the hockey skating actions dataset.

This script inspects the pickle/json files in `/data/skating_actions_dataset`
and prints:
 - Top-level structure and sample contents of the pickle files
 - Skeleton array shapes (frames, joints, coords, persons)
 - Temporal distribution (frame counts)
 - Class balance if labels can be located
"""

import json
import os
import pickle
from collections import Counter, defaultdict
from typing import Any, Dict, Iterable, List, Tuple

import numpy as np


DATA_ROOT = "../../../../data/skating_actions_dataset"
ANNOTATION_DIR = os.path.join(DATA_ROOT, "annotations")
PICKLE_FILES = [
    "train.pkl",
    "val.pkl",
    "test.pkl",
    "combined_annotations.pkl",
]

# How many entries to sample for stats
SAMPLE_LIMIT = 2000
PRINT_SAMPLE_ENTRY = True
SAMPLE_INDEX = 0


def load_pickle(path: str) -> Any:
    with open(path, "rb") as f:
        return pickle.load(f)


def describe_obj(obj: Any, depth: int = 0, max_items: int = 3) -> str:
    """Lightweight recursive describer to get a feel for arbitrary structures."""
    prefix = "  " * depth
    if isinstance(obj, dict):
        keys = list(obj.keys())
        snippet = ", ".join(map(str, keys[:max_items]))
        return f"{prefix}dict(len={len(obj)}): keys=[{snippet}{'...' if len(keys) > max_items else ''}]"
    if isinstance(obj, (list, tuple)):
        desc = f"{prefix}{type(obj).__name__}(len={len(obj)})"
        if obj:
            desc += f"\n{describe_obj(obj[0], depth + 1, max_items)}"
        return desc
    if isinstance(obj, np.ndarray):
        return f"{prefix}ndarray shape={obj.shape} dtype={obj.dtype}"
    return f"{prefix}{type(obj).__name__}: {str(obj)[:120]}"


def extract_sequence_stats(entry: Any) -> Tuple[int, int, int, int]:
    """
    Attempt to infer (T, V, C, M) from a single entry.
    Returns (frames, joints, coords, persons) or (0, 0, 0, 0) if unknown.
    """
    # Common patterns: ndarray already in C,T,V,M format or T,V,C or list
    if isinstance(entry, np.ndarray):
        if entry.ndim == 5:
            # Expected GAP format: N,C,T,V,M but N is squeezed
            _, C, T, V, M = entry.shape
            return T, V, C, M
        if entry.ndim == 4:
            # Could be C,T,V,M or T,V,C,M
            shapes = entry.shape
            # Heuristic: if first dim small (<=3) treat as C
            if shapes[0] <= 4:
                C, T, V, M = shapes
                return T, V, C, M
            else:
                T, V, C, M = shapes
                return T, V, C, M
        if entry.ndim == 3:
            # Possibly T,V,C
            T, V, C = entry.shape
            return T, V, C, 1
    if isinstance(entry, dict):
        for key in ["skeleton", "keypoints", "pose", "coords"]:
            if key in entry:
                return extract_sequence_stats(entry[key])
    if isinstance(entry, (list, tuple)) and entry and isinstance(entry[0], (list, tuple, np.ndarray)):
        try:
            arr = np.array(entry)
            return extract_sequence_stats(arr)
        except Exception:
            pass
    return 0, 0, 0, 0


def extract_label(entry: Any) -> Any:
    """Try to pull a label from a sample entry."""
    if isinstance(entry, dict):
        for key in ["label", "action", "class", "action_id", "category"]:
            if key in entry:
                return entry[key]
    if isinstance(entry, (list, tuple)) and entry and not isinstance(entry[0], (list, tuple, np.ndarray, dict)):
        return entry[-1]
    return None


def summarize_pickle(path: str) -> Dict[str, Any]:
    summary: Dict[str, Any] = {"path": path}
    try:
        obj = load_pickle(path)
    except Exception as e:
        summary["error"] = f"failed to load: {e}"
        return summary

    summary["type"] = type(obj).__name__
    summary["length"] = len(obj) if hasattr(obj, "__len__") else "N/A"
    summary["structure"] = describe_obj(obj)

    # Sample entry inspection
    sample = None
    if isinstance(obj, dict):
        # Grab first item
        try:
            first_key = next(iter(obj.keys()))
            sample = obj[first_key]
        except StopIteration:
            sample = None
    elif isinstance(obj, (list, tuple)):
        if len(obj) > 0:
            sample = obj[0]

    if sample is not None:
        summary["sample_structure"] = describe_obj(sample, depth=1)
        T, V, C, M = extract_sequence_stats(sample)
        if T:
            summary["sample_shape_inferred"] = {"frames": T, "joints": V, "coords": C, "persons": M}
        lbl = extract_label(sample)
        if lbl is not None:
            summary["sample_label"] = lbl

    # Attempt class distribution
    labels: List[Any] = []
    if isinstance(obj, (list, tuple)):
        for entry in obj:
            lbl = extract_label(entry)
            if lbl is not None:
                labels.append(lbl)
    elif isinstance(obj, dict):
        for entry in obj.values():
            lbl = extract_label(entry)
            if lbl is not None:
                labels.append(lbl)

    if labels:
        ctr = Counter(labels)
        summary["class_counts"] = dict(ctr)

    return summary


def summarize_actions_json(path: str) -> Dict[str, Any]:
    summary: Dict[str, Any] = {"path": path}
    try:
        data = json.load(open(path, "r"))
    except Exception as e:
        summary["error"] = f"failed to load: {e}"
        return summary
    summary["length"] = len(data) if isinstance(data, list) else "N/A"
    if isinstance(data, list) and data:
        summary["sample_keys"] = list(data[0].keys())
    return summary


def analyze_temporal_stats(entries: Iterable[Any]) -> Dict[str, float]:
    frames: List[int] = []
    for entry in entries:
        T, _, _, _ = extract_sequence_stats(entry)
        if T:
            frames.append(T)
    if not frames:
        return {}
    return {
        "count": len(frames),
        "min_frames": int(np.min(frames)),
        "max_frames": int(np.max(frames)),
        "mean_frames": float(np.mean(frames)),
        "p25_frames": float(np.percentile(frames, 25)),
        "p50_frames": float(np.percentile(frames, 50)),
        "p75_frames": float(np.percentile(frames, 75)),
    }

def inspect_keypoint_semantics(data_root):
    """
    Attempts to find the semantic meaning of the 20 keypoints.
    """
    print("=== Phase 1.5: Investigating Keypoint Semantics ===\n")
    
    # 1. Check annotations_with_pose.json for COCO-style metadata
    pose_json_path = os.path.join(data_root, "annotations_with_pose.json")
    if os.path.exists(pose_json_path):
        print(f"Loading {pose_json_path} to look for keypoint names...")
        try:
            with open(pose_json_path, 'r') as f:
                data = json.load(f)
            
            # Check for standard COCO 'categories' field
            if 'categories' in data:
                print("Found 'categories' metadata:")
                for cat in data['categories']:
                    if 'keypoints' in cat:
                        kps = cat['keypoints']
                        print(f"  Category: {cat.get('name', 'unknown')}")
                        print(f"  Num Keypoints: {len(kps)}")
                        print(f"  Keypoint Names: {kps}")
                        return # Found it, we are done
            else:
                print("  'categories' field not found in JSON.")
                
        except Exception as e:
            print(f"  Error reading json: {e}")
    else:
        print(f"  {pose_json_path} does not exist.")

    # 2. Heuristic Check using Train Data (Fallback)
    # If we can't find names, we analyze Y-coordinates to guess Head vs Foot
    print("\nPerforming Heuristic Analysis on Train Data...")
    train_pkl = os.path.join(data_root, "annotations/train.pkl")
    try:
        with open(train_pkl, "rb") as f:
            data = pickle.load(f)
            
        # Aggregate all keypoints
        all_kps = []
        # Extract first 100 samples
        samples = data['annotations'][:100]
        for s in samples:
            # keypoint shape is (1, 30, 20, 2) -> take 0th index for person
            kps = s['keypoint'][0] # (30, 20, 2)
            all_kps.append(kps)
            
        all_kps = np.concatenate(all_kps, axis=0) # (Frames, 20, 2)
        
        # Calculate mean Y for each of the 20 joints
        mean_y = np.mean(all_kps[:, :, 1], axis=0)
        
        # Sort indices by Y value (Top of image = low Y, Bottom = high Y)
        sorted_indices = np.argsort(mean_y)
        
        print(f"  Inferred Topology by Y-coordinate (0=Top/Head, >0=Bottom/Feet):")
        print(f"  Top-most joints (Head?): {sorted_indices[:3]}")
        print(f"  Bottom-most joints (Feet?): {sorted_indices[-3:]}")
        print("  *Use this to manually map to standard skeletons (OpenPose/COCO)*")
        
    except Exception as e:
        print(f"  Could not perform heuristic analysis: {e}")

# Add this to your main function call



def main():
    print("=== Phase 1: Hockey dataset structure analysis ===\n")

    # Summarize actions.json
    actions_json = os.path.join(DATA_ROOT, "actions.json")
    if os.path.exists(actions_json):
        actions_summary = summarize_actions_json(actions_json)
        print("[actions.json]")
        for k, v in actions_summary.items():
            print(f"  {k}: {v}")
        print()
    else:
        print("actions.json not found\n")

    # Summarize pickle files
    for fname in PICKLE_FILES:
        path = os.path.join(ANNOTATION_DIR, fname)
        if not os.path.exists(path):
            print(f"[{fname}] missing\n")
            continue
        print(f"[{fname}]")
        summary = summarize_pickle(path)
        for k, v in summary.items():
            if k == "class_counts" and isinstance(v, dict):
                # Print top 20 classes if many
                items = list(v.items())
                items.sort(key=lambda x: x[1], reverse=True)
                top = items[:20]
                print("  class_counts (top):")
                for cls, cnt in top:
                    print(f"    {cls}: {cnt}")
            elif k == "structure":
                print(f"  {k}: {v}")
            else:
                print(f"  {k}: {v}")
        print()

        obj = load_pickle(path)
        # Detailed sample entry print
        if isinstance(obj, dict) and "annotations" in obj and isinstance(obj["annotations"], list):
            ann_list = obj["annotations"]
            if ann_list:
                idx = min(SAMPLE_INDEX, len(ann_list) - 1)
                entry = ann_list[idx]
                print(f"  Sample entry at index {idx}:")
                if isinstance(entry, dict):
                    for key, val in entry.items():
                        if isinstance(val, np.ndarray):
                            print(f"    {key}: ndarray shape={val.shape} dtype={val.dtype}")
                        elif isinstance(val, list):
                            print(f"    {key}: list(len={len(val)}) sample={val[:3]}")
                        else:
                            print(f"    {key}: {val}")
                else:
                    print(f"    {entry}")
                print()

        # Additional temporal stats if object is list/tuple of entries
        entries_for_stats: List[Any] = []
        if isinstance(obj, (list, tuple)):
            entries_for_stats = list(obj[: min(SAMPLE_LIMIT, len(obj))])
        elif isinstance(obj, dict) and "annotations" in obj and isinstance(obj["annotations"], list):
            entries_for_stats = obj["annotations"][: min(SAMPLE_LIMIT, len(obj["annotations"]))]

        if entries_for_stats:
            t_stats = analyze_temporal_stats(entries_for_stats)
            if t_stats:
                print("  Temporal stats (sampled):")
                for k, v in t_stats.items():
                    print(f"    {k}: {v}")
                print()

            # Class balance
            labels = []
            for e in entries_for_stats:
                lbl = extract_label(e)
                if lbl is not None:
                    labels.append(lbl)
            if labels:
                ctr = Counter(labels)
                items = list(ctr.items())
                items.sort(key=lambda x: x[1], reverse=True)
                top = items[:20]
                print("  Class counts (sampled top):")
                for cls, cnt in top:
                    print(f"    {cls}: {cnt}")
                print()
    
    inspect_keypoint_semantics(DATA_ROOT)   

if __name__ == "__main__":
    main()

