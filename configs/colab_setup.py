"""
Colab Setup Script
==================
Run this in the first cell of the Colab notebook to set up the environment.
Mounts Google Drive, installs dependencies, configures Kaggle API, and verifies GPU.

Usage in Colab notebook:
    %run configs/colab_setup.py
"""

import subprocess
import sys
import os


def mount_drive():
    """Mount Google Drive for persistent storage."""
    try:
        from google.colab import drive
        drive.mount('/content/drive')
        print("[✓] Google Drive mounted at /content/drive")

        # Create project directory on Drive for checkpoints and logs
        drive_project = '/content/drive/MyDrive/snn-evidence'
        os.makedirs(drive_project, exist_ok=True)
        os.makedirs(os.path.join(drive_project, 'checkpoints'), exist_ok=True)
        os.makedirs(os.path.join(drive_project, 'logs'), exist_ok=True)
        os.makedirs(os.path.join(drive_project, 'data'), exist_ok=True)
        print(f"[✓] Project directory created at {drive_project}")
        return drive_project
    except ImportError:
        print("[!] Not running in Colab — skipping Drive mount")
        return None


def install_dependencies():
    """Install project dependencies."""
    print("\n[...] Installing dependencies...")

    # Core packages
    packages = [
        "git+https://github.com/fangwei123456/spikingjelly.git",
        "opencv-python>=4.8",
        "ffmpeg-python>=0.2.0",
        "fvcore>=0.1.5",
        "kaggle>=1.5",
        "pyyaml>=6.0",
        "tensorboard>=2.14",
        "scipy>=1.11",
    ]

    for pkg in packages:
        subprocess.check_call(
            [sys.executable, "-m", "pip", "install", "-q", pkg],
            stdout=subprocess.DEVNULL,
            stderr=subprocess.DEVNULL,
        )

    print("[✓] All dependencies installed")

    # Verify SpikingJelly
    try:
        import spikingjelly
        print(f"[✓] SpikingJelly version: {spikingjelly.__version__}")
    except ImportError:
        print("[✗] SpikingJelly import failed!")
        raise


def setup_kaggle():
    """Configure Kaggle API credentials."""
    print("\n[...] Setting up Kaggle API...")

    kaggle_dir = os.path.expanduser("~/.kaggle")
    kaggle_json = os.path.join(kaggle_dir, "kaggle.json")

    if os.path.exists(kaggle_json):
        print(f"[✓] Kaggle credentials found at {kaggle_json}")
        return True

    # Try to find kaggle.json in Drive
    drive_kaggle = "/content/drive/MyDrive/kaggle.json"
    if os.path.exists(drive_kaggle):
        os.makedirs(kaggle_dir, exist_ok=True)
        import shutil
        shutil.copy(drive_kaggle, kaggle_json)
        os.chmod(kaggle_json, 0o600)
        print(f"[✓] Copied kaggle.json from Drive to {kaggle_json}")
        return True

    print("[!] Kaggle credentials not found.")
    print("    Option 1: Upload kaggle.json to Google Drive root")
    print("    Option 2: Set environment variables:")
    print("      import os")
    print('      os.environ["KAGGLE_USERNAME"] = "your_username"')
    print('      os.environ["KAGGLE_KEY"] = "your_api_key"')
    return False


def verify_gpu():
    """Verify GPU availability and print specs."""
    import torch

    print("\n[...] Checking GPU...")
    if torch.cuda.is_available():
        gpu_name = torch.cuda.get_device_name(0)
        gpu_mem = torch.cuda.get_device_properties(0).total_mem / (1024 ** 3)
        print(f"[✓] GPU: {gpu_name} ({gpu_mem:.1f} GB)")
        print(f"[✓] CUDA version: {torch.version.cuda}")
        print(f"[✓] PyTorch version: {torch.__version__}")

        # Check if memory is sufficient for batch_size=16, 112x112
        if gpu_mem >= 14:
            print("[✓] Memory sufficient for batch_size=16")
        elif gpu_mem >= 8:
            print("[!] Consider reducing batch_size to 8")
        else:
            print("[!] Limited GPU memory — use batch_size=4")
    else:
        print("[✗] No GPU detected!")
        print("    Go to Runtime → Change runtime type → GPU")

    return torch.cuda.is_available()


def setup_project_symlinks(drive_project):
    """Create symlinks from project dirs to Drive for persistence."""
    if drive_project is None:
        return

    project_root = os.getcwd()

    # Symlink results dir to Drive
    results_local = os.path.join(project_root, "results")
    results_drive = os.path.join(drive_project, "results")
    os.makedirs(results_drive, exist_ok=True)

    if not os.path.exists(results_local):
        os.symlink(results_drive, results_local)
        print(f"[✓] Symlinked results/ → {results_drive}")

    # Symlink data/raw to Drive (avoid re-downloading)
    data_drive = os.path.join(drive_project, "data")
    data_local = os.path.join(project_root, "data", "raw")
    os.makedirs(data_drive, exist_ok=True)
    os.makedirs(os.path.join(project_root, "data"), exist_ok=True)

    if not os.path.exists(data_local):
        os.symlink(data_drive, data_local)
        print(f"[✓] Symlinked data/raw/ → {data_drive}")


# ---- Run setup ----
if __name__ == "__main__" or "google.colab" in sys.modules:
    print("=" * 60)
    print("  SNN Violence Detection — Colab Setup")
    print("=" * 60)

    drive_project = mount_drive()
    install_dependencies()
    setup_kaggle()
    gpu_ok = verify_gpu()
    setup_project_symlinks(drive_project)

    print("\n" + "=" * 60)
    if gpu_ok:
        print("  Setup complete! Ready to train.")
    else:
        print("  Setup complete, but no GPU detected.")
    print("=" * 60)
