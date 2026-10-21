"""
VisionGuard SNN - Automated RLVS Dataset Downloader & Extractor
=============================================================
Downloads the Real-Life Violence Situations (RLVS) dataset (~2.88 GB)
from the direct Hugging Face repository, verifies integrity, extracts videos,
and structures it inside data/raw/rlvs.
"""

import os
import sys
import time
import zipfile
import urllib.request
from pathlib import Path

DATA_DIR = Path(__file__).resolve().parent.parent / "data"
RAW_DIR = DATA_DIR / "raw" / "rlvs"
ZIP_PATH = DATA_DIR / "Real_Life_Violence_Dataset.zip"
URL = "https://huggingface.co/datasets/khoipd/Violence/resolve/main/Real_Life_Violence_Dataset.zip"

def download_with_progress(url: str, output_path: Path):
    output_path.parent.mkdir(parents=True, exist_ok=True)
    temp_path = output_path.with_suffix(".tmp")
    
    downloaded = 0
    if temp_path.exists():
        downloaded = temp_path.stat().st_size
        print(f"[RLVS Downloader] Resuming download from {downloaded / (1024**2):.1f} MB...")
        
    req = urllib.request.Request(url)
    if downloaded > 0:
        req.add_header("Range", f"bytes={downloaded}-")
        
    print(f"[RLVS Downloader] Connecting to {url}...")
    with urllib.request.urlopen(req, timeout=30) as resp:
        total_len = resp.headers.get("Content-Length")
        total_size = int(total_len) + downloaded if total_len else None
        
        mode = "ab" if downloaded > 0 else "wb"
        start_time = time.time()
        last_log = time.time()
        
        with open(temp_path, mode) as f:
            chunk_size = 1024 * 1024 # 1 MB chunks
            while True:
                chunk = resp.read(chunk_size)
                if not chunk:
                    break
                f.write(chunk)
                downloaded += len(chunk)
                
                # Log progress every 5 seconds
                if time.time() - last_log > 5.0:
                    last_log = time.time()
                    elapsed = time.time() - start_time
                    speed_mb = (downloaded / (1024**2)) / max(1.0, elapsed)
                    if total_size:
                        pct = (downloaded / total_size) * 100
                        print(f"[RLVS Downloader] {downloaded / (1024**2):.1f} MB / {total_size / (1024**2):.1f} MB ({pct:.1f}%) | {speed_mb:.2f} MB/s")
                    else:
                        print(f"[RLVS Downloader] {downloaded / (1024**2):.1f} MB downloaded | {speed_mb:.2f} MB/s")

    if temp_path.exists():
        temp_path.rename(output_path)
    print(f"[RLVS Downloader] Download completed successfully: {output_path} ({output_path.stat().st_size / (1024**2):.1f} MB)")


def extract_rlvs(zip_path: Path, extract_dir: Path):
    print(f"[RLVS Extractor] Extracting {zip_path} to {extract_dir}...")
    extract_dir.mkdir(parents=True, exist_ok=True)
    
    with zipfile.ZipFile(zip_path, 'r') as zip_ref:
        total_files = len(zip_ref.infolist())
        print(f"[RLVS Extractor] Found {total_files} files inside archive. Extracting...")
        
        extracted = 0
        last_log = time.time()
        for member in zip_ref.infolist():
            zip_ref.extract(member, extract_dir)
            extracted += 1
            if time.time() - last_log > 5.0:
                last_log = time.time()
                pct = (extracted / total_files) * 100
                print(f"[RLVS Extractor] Progress: {extracted}/{total_files} ({pct:.1f}%)")
                
    print("[RLVS Extractor] Extraction complete!")


def organize_and_verify(extract_dir: Path):
    print("[RLVS Verifier] Inspecting extracted structure...")
    all_videos = list(extract_dir.rglob("*.mp4")) + list(extract_dir.rglob("*.avi"))
    print(f"[RLVS Verifier] Total video files found: {len(all_videos)}")
    
    violence_vids = [v for v in all_videos if "violence" in v.parent.name.lower() and "non" not in v.parent.name.lower()]
    non_violence_vids = [v for v in all_videos if "non" in v.parent.name.lower()]
    
    print(f"[RLVS Verifier] Violence videos: {len(violence_vids)}")
    print(f"[RLVS Verifier] Non-Violence videos: {len(non_violence_vids)}")
    
    return len(all_videos) > 0


if __name__ == "__main__":
    if not ZIP_PATH.exists() or ZIP_PATH.stat().st_size < 1024 * 1024 * 100:
        download_with_progress(URL, ZIP_PATH)
    else:
        print(f"[RLVS Downloader] Archive already downloaded: {ZIP_PATH} ({ZIP_PATH.stat().st_size / (1024**2):.1f} MB)")
        
    extract_rlvs(ZIP_PATH, RAW_DIR)
    success = organize_and_verify(RAW_DIR)
    if success:
        print("\n[SUCCESS] RLVS dataset downloaded, extracted, and verified successfully!")
    else:
        print("\n[WARNING] Extraction completed but video count unexpected.")
