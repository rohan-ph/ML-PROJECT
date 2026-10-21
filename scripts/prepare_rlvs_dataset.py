"""
VisionGuard SNN - RLVS Dataset Preprocessor & Splitter
=====================================================
Processes raw RLVS videos into fast, memory-mapped temporal difference tensors [16, 3, 112, 112]
with reproducible 70% Train, 15% Val, 15% Test splits (Seed 42).
"""

import os
import sys
import time
import json
import random
import pathlib
import cv2
import numpy as np
import concurrent.futures

BASE_DIR = pathlib.Path(__file__).resolve().parent.parent
RAW_DIR = BASE_DIR / "data" / "raw" / "rlvs"
PROCESSED_DIR = BASE_DIR / "data" / "processed" / "rlvs"
SPLITS_FILE = BASE_DIR / "data" / "rlvs_splits.json"

TARGET_FRAMES = 16
IMAGE_SIZE = 112
SEED = 42

def process_single_video(args):
    src_path, dest_path, cls_label = args
    if dest_path.exists() and dest_path.stat().st_size > 1000:
        return True, src_path.name, "cached"
        
    dest_path.parent.mkdir(parents=True, exist_ok=True)
    cap = cv2.VideoCapture(str(src_path))
    total_frames = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))
    
    if total_frames < 2:
        cap.release()
        return False, src_path.name, "too_short"
        
    # Sample TARGET_FRAMES + 1 equidistant frames to compute TARGET_FRAMES differences
    sample_indices = np.linspace(0, max(0, total_frames - 1), TARGET_FRAMES + 1, dtype=int)
    idx_set = set(sample_indices)
    
    sampled_frames = []
    curr_idx = 0
    while cap.isOpened() and len(sampled_frames) < TARGET_FRAMES + 1:
        ret, frame = cap.read()
        if not ret:
            break
        if curr_idx in idx_set:
            h, w = frame.shape[:2]
            min_dim = min(h, w)
            sy = (h - min_dim) // 2
            sx = (w - min_dim) // 2
            square_crop = frame[sy:sy+min_dim, sx:sx+min_dim]
            resized = cv2.resize(square_crop, (IMAGE_SIZE, IMAGE_SIZE))
            sampled_frames.append(resized)
        curr_idx += 1
    cap.release()
    
    if len(sampled_frames) < 2:
        return False, src_path.name, "corrupt"
        
    # Pad if needed
    while len(sampled_frames) < TARGET_FRAMES + 1:
        sampled_frames.append(sampled_frames[-1])
        
    # Compute frame differences
    diffs = []
    for i in range(TARGET_FRAMES):
        diff = cv2.absdiff(sampled_frames[i+1], sampled_frames[i])
        diff_rgb = cv2.cvtColor(diff, cv2.COLOR_BGR2RGB)
        diffs.append(diff_rgb)
        
    arr = np.stack(diffs, axis=0) # [16, 112, 112, 3]
    arr = np.transpose(arr, (0, 3, 1, 2)) # [16, 3, 112, 112]
    arr_uint8 = arr.astype(np.uint8)
    
    np.save(str(dest_path), arr_uint8)
    return True, src_path.name, "processed"


def main():
    print("=" * 60)
    print(" VisionGuard SNN - RLVS Dataset Preprocessing & Splitting ")
    print("=" * 60)
    
    random.seed(SEED)
    np.random.seed(SEED)
    
    # 1. Collect all video files
    v_files = sorted(list((RAW_DIR / "Violence").glob("*.mp4")) + list((RAW_DIR / "Violence").glob("*.avi")))
    nv_files = sorted(list((RAW_DIR / "NonViolence").glob("*.mp4")) + list((RAW_DIR / "NonViolence").glob("*.avi")))
    
    print(f"Discovered raw files: {len(v_files)} Violence, {len(nv_files)} NonViolence (Total: {len(v_files) + len(nv_files)})")
    
    # 2. Shuffle reproducibly
    random.shuffle(v_files)
    random.shuffle(nv_files)
    
    def split(files):
        n = len(files)
        n_train = int(n * 0.70)
        n_val = int(n * 0.15)
        return files[:n_train], files[n_train:n_train+n_val], files[n_train+n_val:]
        
    v_tr, v_val, v_te = split(v_files)
    nv_tr, nv_val, nv_te = split(nv_files)
    
    splits_manifest = {
        "seed": SEED,
        "counts": {
            "train": {"violence": len(v_tr), "non_violence": len(nv_tr), "total": len(v_tr) + len(nv_tr)},
            "val": {"violence": len(v_val), "non_violence": len(nv_val), "total": len(v_val) + len(nv_val)},
            "test": {"violence": len(v_te), "non_violence": len(nv_te), "total": len(v_te) + len(nv_te)}
        },
        "files": {
            "train": [str(f.relative_to(RAW_DIR)) for f in v_tr + nv_tr],
            "val": [str(f.relative_to(RAW_DIR)) for f in v_val + nv_val],
            "test": [str(f.relative_to(RAW_DIR)) for f in v_te + nv_te]
        }
    }
    
    with open(SPLITS_FILE, "w") as f:
        json.dump(splits_manifest, f, indent=2)
    print(f"Splits JSON saved to: {SPLITS_FILE}")
    print(f"  • Train: {len(v_tr)+len(nv_tr)} samples (70%)")
    print(f"  • Val:   {len(v_val)+len(nv_val)} samples (15%)")
    print(f"  • Test:  {len(v_te)+len(nv_te)} samples (15%)")
    
    # 3. Build tasks
    tasks = []
    for split_name, v_list, nv_list in [("train", v_tr, nv_tr), ("val", v_val, nv_val), ("test", v_te, nv_te)]:
        for v in v_list:
            dest = PROCESSED_DIR / split_name / "Violence" / f"{v.stem}.npy"
            tasks.append((v, dest, 1))
        for nv in nv_list:
            dest = PROCESSED_DIR / split_name / "NonViolence" / f"{nv.stem}.npy"
            tasks.append((nv, dest, 0))
            
    print(f"\nPreprocessing {len(tasks)} videos into multi-step frame difference arrays [16, 3, 112, 112]...")
    start_t = time.time()
    
    success_count = 0
    fail_count = 0
    cached_count = 0
    
    # Process in parallel with ThreadPoolExecutor
    workers = min(12, os.cpu_count() or 4)
    with concurrent.futures.ThreadPoolExecutor(max_workers=workers) as executor:
        for i, (ok, fname, status) in enumerate(executor.map(process_single_video, tasks)):
            if ok:
                if status == "cached":
                    cached_count += 1
                else:
                    success_count += 1
            else:
                fail_count += 1
                
            if (i + 1) % 200 == 0 or (i + 1) == len(tasks):
                elapsed = time.time() - start_t
                rate = (i + 1) / max(1.0, elapsed)
                pct = ((i + 1) / len(tasks)) * 100
                print(f"  [{i+1}/{len(tasks)}] {pct:.1f}% processed ({rate:.1f} vids/sec) - {fail_count} failed")
                
    total_time = time.time() - start_t
    print(f"\n[DONE] Preprocessing completed in {total_time:.1f}s ({total_time/60:.1f} min)")
    print(f"  • Newly processed: {success_count}")
    print(f"  • Already cached:  {cached_count}")
    print(f"  • Corrupted/failed: {fail_count}")

if __name__ == "__main__":
    main()
