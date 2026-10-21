"""
Dataset downloader script for the SNN violence detection project.
Downloads RWF-2000, Hockey Fight, and RLVS datasets using the Kaggle API.
"""

import os
import argparse
import subprocess
import glob
from pathlib import Path
import logging

logging.basicConfig(level=logging.INFO, format="%(asctime)s - %(levelname)s - %(message)s")

DATASETS = {
    "rwf2000": "vulamnguyen/rwf2000",
    "hockey": "yassershrief/hockey-fight-vidoes",
    "rlvs": "mohamedmustafa/real-life-violence-situations-dataset"
}

def check_kaggle_api():
    """Verify Kaggle API key is accessible."""
    kaggle_json = Path.home() / ".kaggle" / "kaggle.json"
    if not kaggle_json.exists():
        logging.warning("Kaggle API key not found at ~/.kaggle/kaggle.json")
        logging.warning("Please ensure Kaggle API is configured to download datasets.")
        return False
    return True

def download_dataset(dataset_key: str, data_dir: str):
    """Download and extract a dataset using the Kaggle CLI."""
    if dataset_key not in DATASETS:
        raise ValueError(f"Unknown dataset: {dataset_key}")
        
    kaggle_path = DATASETS[dataset_key]
    target_dir = Path(data_dir) / "raw" / dataset_key
    
    if target_dir.exists() and any(target_dir.iterdir()):
        logging.info(f"Dataset {dataset_key} seems to be already downloaded at {target_dir}")
        return
        
    target_dir.mkdir(parents=True, exist_ok=True)
    logging.info(f"Downloading {dataset_key} to {target_dir}...")
    
    try:
        subprocess.run(
            ["kaggle", "datasets", "download", "-d", kaggle_path, "-p", str(target_dir), "--unzip"],
            check=True
        )
        logging.info(f"Successfully downloaded and extracted {dataset_key}")
    except subprocess.CalledProcessError as e:
        logging.error(f"Failed to download {dataset_key}. Is kaggle CLI installed and configured?")
        raise e
    except FileNotFoundError:
        logging.error("Kaggle CLI not found. Please run: pip install kaggle")
        raise

def validate_dataset(dataset_key: str, data_dir: str):
    """Validate expected file counts after extraction."""
    target_dir = Path(data_dir) / "raw" / dataset_key
    
    if not target_dir.exists():
        logging.error(f"Directory {target_dir} does not exist.")
        return

    # Count video files (avi, mp4)
    videos = list(target_dir.rglob("*.avi")) + list(target_dir.rglob("*.mp4"))
    logging.info(f"Found {len(videos)} video files for {dataset_key}.")
    
    if dataset_key == "rwf2000" and len(videos) != 2000:
        logging.warning(f"Expected 2000 files for RWF-2000, found {len(videos)}")
    elif dataset_key == "hockey" and len(videos) != 1000:
        logging.warning(f"Expected 1000 files for Hockey Fight, found {len(videos)}")
    elif dataset_key == "rlvs" and len(videos) != 2000:
        logging.warning(f"Expected 2000 files for RLVS, found {len(videos)}")

def main():
    parser = argparse.ArgumentParser(description="Download video datasets for violence detection.")
    parser.add_argument("--dataset", choices=["rwf2000", "hockey", "rlvs", "all"], default="all",
                        help="Dataset to download")
    parser.add_argument("--data-dir", type=str, default="d:/projectzz/ML PROJ/snn-evidence/data",
                        help="Base directory for data")
    args = parser.parse_args()

    check_kaggle_api()
    
    datasets_to_download = list(DATASETS.keys()) if args.dataset == "all" else [args.dataset]
    
    for ds in datasets_to_download:
        download_dataset(ds, args.data_dir)
        validate_dataset(ds, args.data_dir)

if __name__ == "__main__":
    main()
