"""
Synthetic surveillance stream generator.
Concatenates processed clips into long streams with violence making up <5%.
Records ground-truth onset/offset timestamps in a JSON annotation file.
"""

import os
import argparse
import random
import json
import yaml
import numpy as np
from pathlib import Path
from tqdm import tqdm
import logging

logging.basicConfig(level=logging.INFO, format="%(asctime)s - %(levelname)s - %(message)s")

def load_clip_paths(data_dir: Path, dataset: str = "rwf2000", encoding: str = "diff"):
    """Load paths for fight and non-fight clips."""
    base_dir = data_dir / "processed" / dataset
    
    # RWF-2000 has train/val splits and Fight/NonFight subdirs
    # We will just collect all available for stream generation
    fight_paths = list(base_dir.rglob(f"Fight/*_{encoding}.npy"))
    nonfight_paths = list(base_dir.rglob(f"NonFight/*_{encoding}.npy"))
    
    return fight_paths, nonfight_paths

def make_stream(
    stream_idx: int,
    out_dir: Path,
    fight_paths: list,
    nonfight_paths: list,
    fps: int,
    min_duration_sec: int,
    max_duration_sec: int,
    violence_ratio: float,
    min_gap_sec: int
):
    """Generate a single stream and save as .npy and .json."""
    duration_sec = random.randint(min_duration_sec, max_duration_sec)
    total_frames = duration_sec * fps
    
    # Target number of violence frames
    target_violence_frames = int(total_frames * violence_ratio)
    
    current_frames = 0
    stream_data = []
    annotations = []
    
    # To handle "Violence at stream start" edge case, we might randomly force one at start
    force_start_violence = random.random() < 0.1  # 10% chance
    
    last_violence_end_frame = - (min_gap_sec * fps) # ensure we can place at start
    
    while current_frames < total_frames:
        frames_left = total_frames - current_frames
        
        # Decide whether to insert violence or non-violence
        insert_violence = False
        
        # If we need more violence and the gap requirement is satisfied
        if target_violence_frames > 0 and (current_frames - last_violence_end_frame) >= (min_gap_sec * fps):
            # Probability to insert violence increases if we are falling behind
            # or force at start
            if current_frames == 0 and force_start_violence:
                insert_violence = True
            elif random.random() < 0.1: # 10% chance to insert violence if conditions met
                insert_violence = True
        
        # If we are almost at the end and no violence was inserted, control stream is formed.
        if insert_violence and fight_paths:
            clip_path = random.choice(fight_paths)
            is_fight = True
        elif nonfight_paths:
            clip_path = random.choice(nonfight_paths)
            is_fight = False
        else:
            break # No paths available
            
        clip = np.load(clip_path) # shape: [T, C, H, W]
        T_clip = clip.shape[0]
        
        # Truncate if clip exceeds remaining stream length
        if T_clip > frames_left:
            clip = clip[:frames_left]
            T_clip = frames_left
            
        stream_data.append(clip)
        
        if is_fight:
            annotations.append({
                "start_frame": current_frames,
                "end_frame": current_frames + T_clip - 1,
                "clip_path": str(clip_path)
            })
            last_violence_end_frame = current_frames + T_clip - 1
            target_violence_frames -= T_clip
            
        current_frames += T_clip
        
    if not stream_data:
        return
        
    full_stream = np.concatenate(stream_data, axis=0) # [Total_T, C, H, W]
    
    npy_path = out_dir / f"stream_{stream_idx:04d}.npy"
    json_path = out_dir / f"stream_{stream_idx:04d}.json"
    
    # Save large file
    np.save(str(npy_path), full_stream)
    
    # Save annotations
    anno_data = {
        "stream_id": stream_idx,
        "duration_sec": duration_sec,
        "total_frames": int(full_stream.shape[0]),
        "fps": fps,
        "incidents": annotations
    }
    with open(json_path, 'w') as f:
        json.dump(anno_data, f, indent=4)
        
    logging.info(f"Stream {stream_idx} created: {full_stream.shape[0]} frames, {len(annotations)} incidents.")

def generate_synthetic_stream(out_dir: str, num_frames: int = 30):
    """Helper function to generate dummy synthetic stream for smoke testing."""
    os.makedirs(out_dir, exist_ok=True)
    frames = []
    for i in range(num_frames):
        fname = os.path.join(out_dir, f"frame_{i:04d}.png")
        # Touch empty/dummy file
        with open(fname, "wb") as f:
            f.write(b"0"*100)
        frames.append(fname)
    anno_path = os.path.join(out_dir, "annotations.json")
    with open(anno_path, "w") as f:
        json.dump({"incidents": [{"start_frame": 10, "end_frame": 20}]}, f, indent=4)
    return frames, anno_path

def main():
    parser = argparse.ArgumentParser(description="Generate synthetic streams.")
    parser.add_argument("--config", type=str, default="configs/default.yaml", help="Path to config file")
    parser.add_argument("--data-dir", type=str, default="d:/projectzz/ML PROJ/snn-evidence/data", help="Data directory")
    args = parser.parse_args()
    
    if os.path.exists(args.config):
        with open(args.config, 'r') as f:
            config = yaml.safe_load(f)
    else:
        logging.warning(f"Config {args.config} not found, using defaults.")
        config = {}
        
    stream_cfg = config.get("stream_generation", {})
    num_streams = stream_cfg.get("num_streams", 10)
    min_duration = stream_cfg.get("min_duration_sec", 300) # 5 mins
    max_duration = stream_cfg.get("max_duration_sec", 1800) # 30 mins
    violence_ratio = stream_cfg.get("violence_ratio", 0.05)
    min_gap = stream_cfg.get("min_gap_between_incidents_sec", 10)
    fps = stream_cfg.get("fps", 16)
    
    data_dir = Path(args.data_dir)
    out_dir = data_dir / "streams"
    out_dir.mkdir(parents=True, exist_ok=True)
    
    fight_paths, nonfight_paths = load_clip_paths(data_dir, dataset="rwf2000", encoding="diff")
    
    if not fight_paths or not nonfight_paths:
        logging.error("Could not find processed clips. Please run prep_rwf.py first.")
        return
        
    logging.info(f"Generating {num_streams} streams...")
    for i in tqdm(range(num_streams)):
        make_stream(
            i, out_dir, fight_paths, nonfight_paths, fps,
            min_duration, max_duration, violence_ratio, min_gap
        )

if __name__ == "__main__":
    main()
