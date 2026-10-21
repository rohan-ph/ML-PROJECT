# VisionGuard SNN: Real-Time Violence Detection, Neural Spike Visualization & Automated Forensic Evidence Dashboard

[![PyTorch](https://img.shields.io/badge/PyTorch-2.6.0%2Bcu124-EE4C2C.svg)](https://pytorch.org/)
[![CUDA](https://img.shields.io/badge/CUDA-12.4%20(RTX%203050%206GB)-76B900.svg)](https://developer.nvidia.com/cuda-zone)
[![SpikingJelly](https://img.shields.io/badge/SpikingJelly-2.0.0-blue.svg)](https://github.com/fangwei123456/spikingjelly)
[![FastAPI](https://img.shields.io/badge/FastAPI-0.115%2B-009688.svg)](https://fastapi.tiangolo.com/)
[![Tests](https://img.shields.io/badge/Tests-8%2F8%20Passed-brightgreen.svg)]()

> **VisionGuard SNN** is an end-to-end neuromorphic computer vision system that performs real-time violence detection using a biologically-grounded **Spiking Neural Network (SNN)**, extracts temporal boundaries of violent incidents via a leaky integrate-and-fire **IncidentNeuron with Schmitt trigger hysteresis**, automatically preserves forensic evidence clips with exact 10-second contextual margins, and provides a full-stack, glassmorphic telemetry dashboard.

---

## Table of Contents
1. [Key Features](#key-features)
2. [System Architecture](#system-architecture)
3. [Neuromorphic Advantage & Energy Efficiency](#neuromorphic-advantage--energy-efficiency)
4. [Hardware & Environment Setup](#hardware--environment-setup)
5. [Trained Model & Artifacts](#trained-model--artifacts)
6. [Executed Jupyter Notebook](#executed-jupyter-notebook)
7. [Full-Stack Dashboard & Real-Time Telemetry](#full-stack-dashboard--real-time-telemetry)
8. [Dataset Provenance & Benchmark Fixtures](#dataset-provenance--benchmark-fixtures)
9. [Automated Testing & Verification](#automated-testing--verification)
10. [REST & WebSocket API Reference](#rest--websocket-api-reference)

---

## Key Features

- **Genuine Spiking Neural Network (SNN)**: Built on PyTorch and SpikingJelly with 4 convolutional LIF stages and an LIF readout layer using surrogate gradient backpropagation ($\text{ATan}(\alpha=2.0)$) and single-step streaming support.
- **IncidentNeuron with Schmitt Trigger**: Solves frame-level classification jitter via leaky temporal integration ($\beta = 0.90$) and dual-threshold hysteresis ($\theta_{\text{on}} = 0.60, \theta_{\text{off}} = 0.30$) with minimum duration (8 frames) and cooldown (16 frames).
- **Exact Forensic Evidence Extraction Guarantee**:
  $$\text{Clip Interval} = \left[ \max(0, t_{\text{start}} - 10\,\text{s}), \; \min(T, t_{\text{end}} + 10\,\text{s}) \right]$$
  Boundary clamping ensures complete safety for early onset ($t_{\text{start}} < 10\,\text{s}$) and late onset ($t_{\text{end}} + 10\,\text{s} > T$).
- **Cryptographic SHA-256 Forensics**: Every extracted clip is automatically hashed upon disk write, re-read independently, and cryptographically verified against tampering before being added to the evidence ledger.
- **Real-Time Spike Telemetry**: Exposes authentic per-layer firing rates, spike counts, readout membrane potential ($V_{\text{mem}}$), and 16 probe neuron raster states streamed at 16 FPS.
- **Modern Glassmorphic Dashboard**: FastAPI backend with WebSocket broadcast paired with a dark-mode frontend featuring real-time Chart.js membrane potential tracking, dynamic spike raster canvas, and direct HTML5 video evidence playback cards.

---

## System Architecture

```text
[ Video Stream / Camera ]
         │
         ▼  (16 FPS)
[ Frame Difference Engine ]  --> ΔI_t = |Frame_t - Frame_{t-1}|
         │
         ▼  [1, 3, 112, 112]
┌────────────────────────────────────────────────────────┐
│               SpikingViolenceNet (SNN)                 │
│                                                        │
│  Conv1 (32ch) ──► LIF0 (τ=2.0) ──► MaxPool (56x56)     │
│  Conv2 (64ch) ──► LIF1 (τ=2.0) ──► MaxPool (28x28)     │
│  Conv3 (128ch)──► LIF2 (τ=2.0) ──► MaxPool (14x14)     │
│  Conv4 (256ch)──► LIF3 (τ=2.0) ──► MaxPool (7x7)       │
│                                                        │
│  Global Average Pooling (256) ──► Linear(256, 2)       │
│  Readout LIF (τ=4.0) ──► [V_mem_0, V_mem_1]            │
└────────────────────────────────┬───────────────────────┘
                                 │
           ┌─────────────────────┴─────────────────────┐
           ▼                                           ▼
[ RealTimeSpikeMonitor ]                     [ Softmax / Sigmoid ]
- 16-Neuron Raster Hooks                               │ Score s_t ∈ [0, 1]
- Layer Firing Rates                                   ▼
- Membrane Potentials                    ┌───────────────────────────┐
           │                             │      IncidentNeuron       │
           │                             │  u_t = β u_{t-1} + (1-β) s│
           │                             │  Schmitt Trigger [0.3,0.6]│
           │                             └─────────────┬─────────────┘
           │                                           │ State Event
           │                                           ▼
           │                             ┌───────────────────────────┐
           │                             │       ClipAssembler       │
           │                             │  Ring Buffer (10s margin) │
           │                             │  SHA-256 Forensic Hash    │
           │                             └─────────────┬─────────────┘
           ▼                                           ▼
┌────────────────────────────────────────────────────────────────────┐
│                    FastAPI WebSocket & REST                        │
│            http://127.0.0.1:8000 (Glassmorphic HUD)                │
└────────────────────────────────────────────────────────────────────┘
```

---

## Neuromorphic Advantage & Energy Efficiency

Traditional deep learning models (ANNs, 3D-CNNs, Transformers) perform continuous Multiply-Accumulate (MAC) operations on dense floating-point values for every pixel and channel:

$$E_{\text{ANN}} = N_{\text{MAC}} \times E_{\text{MAC}} \quad \text{where } E_{\text{MAC}} \approx 4.6\,\text{pJ (45nm CMOS)}$$

In contrast, our event-driven Spiking Neural Network only computes sparse Accumulate (AC) additions when a neuron crosses its threshold $\theta$ and emits an action potential:

$$E_{\text{SNN}} = N_{\text{AC}} \times E_{\text{AC}} \quad \text{where } E_{\text{AC}} \approx 0.9\,\text{pJ (45nm CMOS)}$$

### Energy Comparison on 16-Frame Benchmark (112x112 Input)

| Model Architecture | Operation Type | Operations / Clip | Energy per Clip | Energy Reduction |
| :--- | :--- | :--- | :--- | :--- |
| **Standard CNN (ResNet-18)** | Dense MAC | $1.82 \times 10^9$ FLOPs | **$8.37\,\text{mJ}$** | Baseline ($1.0\times$) |
| **MobileNetV2** | Dense MAC | $3.10 \times 10^8$ FLOPs | **$1.43\,\text{mJ}$** | $5.9\times$ |
| **VisionGuard SNN (Ours)** | **Sparse AC** | **$1.35 \times 10^8$ Spikes** | **$0.122\,\text{mJ}$** | **$68.6\times$ vs ResNet-18** ($11.7\times$ vs MobileNet) |

*Calculated via `energy.py` using per-layer synaptic firing counts.*

---

## Hardware & Environment Setup

### System Specifications
- **GPU**: NVIDIA GeForce RTX 3050 6GB Laptop GPU
- **CUDA Version**: 12.4
- **Driver Version**: 581.86
- **Python**: 3.12.0 (Windows 64-bit)

### Installation Steps

1. **Clone the Repository**:
   ```powershell
   git clone <repo-url>
   cd "ML PROJECT FILE"
   ```

2. **Install PyTorch with CUDA 12.4 Support**:
   ```powershell
   pip install torch torchvision --index-url https://download.pytorch.org/whl/cu124
   ```

3. **Install SpikingJelly from GitHub Master** *(avoids NumPy 1.24+ `np.int` deprecation bug)*:
   ```powershell
   pip install git+https://github.com/fangwei123456/spikingjelly.git
   ```

4. **Install Remaining Requirements**:
   ```powershell
   pip install -r requirements.txt
   ```

5. **Verify GPU Acceleration**:
   ```powershell
   python -c "import torch; print('CUDA Available:', torch.cuda.is_available(), '| GPU:', torch.cuda.get_device_name(0))"
   ```

---

## Trained Model & Artifacts

- **Model Checkpoint**: `results/checkpoints/snn_best.pth`
- **Model Manifest**: `results/model_manifest.json`
- **Architecture**: 4x Conv-BN-LIF ($32 \to 64 \to 128 \to 256$) + Readout LIF Node
- **Training Strategy**: Backpropagation Through Time (BPTT) with ATan surrogate gradient ($\alpha=2.0$), Adam optimizer ($\text{lr}=10^{-3}$, cosine annealing to $10^{-5}$), and temporal reset via `functional.reset_net()`.

---

## Executed Jupyter Notebook

The full training and evaluation notebook is located at:
📁 `notebooks/SNN_Violence_Detection_RTX3050.ipynb`

It is **100% pre-executed** on the local RTX 3050 GPU and contains:
1. **System & CUDA 12.4 Verification**: GPU hardware details, memory footprint, PyTorch and SpikingJelly versions.
2. **Dataset Manifest & Inspection**: Verification of benchmark video fixtures and class distributions.
3. **SNN Architecture Definition**: Visual LIF cell diagram, hook registration, and single/multi-step modes.
4. **Pre-Training Spike Raster Inspection**: Untrained baseline showing raw Poisson-like firing noise.
5. **GPU Training Loop**: Real training progression over epochs with loss curves and validation accuracy.
6. **Post-Training Spike Raster & Confusion Matrix**: Organized post-training raster showing temporal divergence between violent bursts and peaceful motions.
7. **IncidentNeuron Hysteresis & Clipping**: Interactive demonstration of $[ \max(0, t_{\text{start}}-10), \min(T, t_{\text{end}}+10) ]$ on standard (60s) and early-onset (20s) streams.
8. **Forensic Evidence Verification**: Extraction of evidence MP4 files and live SHA-256 hash checking.
9. **Energy & Neuromorphic Efficiency Analysis**: Quantitative comparison table ($E_{\text{MAC}}$ vs $E_{\text{AC}}$).

To view or re-execute the notebook:
```powershell
jupyter notebook notebooks/SNN_Violence_Detection_RTX3050.ipynb
```

---

## Full-Stack Dashboard & Real-Time Telemetry

The repository includes a modern web dashboard built with **FastAPI**, **WebSockets**, and **Chart.js**:

### Launching the Dashboard

```powershell
python -m uvicorn dashboard.app:app --host 127.0.0.1 --port 8000
```
Open your browser at **[http://127.0.0.1:8000](http://127.0.0.1:8000)**.

### Dashboard Capabilities
- **System Telemetry Badge**: Real-time status indicator showing active GPU model, allocated VRAM, and PyTorch version.
- **Live Video HUD**: Streams frame difference preprocessing directly to an HTML5 canvas overlay.
- **Membrane Potential Chart**: Plots Readout $V_{\text{mem}}$ (Normal vs. Violence) and Incident Integrator $u_t$ over time.
- **Layer Firing Rate Bar Chart**: Displays live per-layer spike densities across `lif0`, `lif1`, `lif2`, `lif3`, and `readout_lif`.
- **16-Probe Spike Raster**: Canvas visualizer rendering individual action potentials across 16 monitored probe neurons.
- **Evidence Clips Ledger**: Interactive gallery of extracted incident clips with:
  - Incident time window $[t_{\text{start}}, t_{\text{end}}]$ and clip interval $[t_{\text{clip\_start}}, t_{\text{clip\_end}}]$.
  - SHA-256 cryptographic checksum badge with instant status verification.
  - In-browser HTML5 video player modal.
- **Video Upload**: Allows testing custom user video files against the trained SNN pipeline.

---

## Dataset Provenance & Benchmark Fixtures

- **Controlled Synthetic Benchmark Streams**:
  - `data/incident_timing/videos/stream_fixture_60s.mp4`: 60-second video @ 16 FPS with an injected high-frequency violence event at $[30.0\,\text{s}, 35.0\,\text{s}]$. Expected clip: $[20.0\,\text{s}, 45.0\,\text{s}]$ ($25.0\,\text{s}$ duration).
  - `data/incident_timing/videos/stream_fixture_20s_early.mp4`: 20-second video @ 16 FPS with an early violence event at $[3.0\,\text{s}, 8.0\,\text{s}]$. Expected clip: $[0.0\,\text{s}, 18.0\,\text{s}]$ ($18.0\,\text{s}$ duration, clamped at $t=0$).
  - Ground truth labels: `data/incident_timing/annotations.csv`.
- **Public Datasets (RWF-2000, Hockey Fight, RLVS)**:
  - Automated downloader script: `scripts/download_datasets.py`.
  - Dataset provenance manifest: `data/dataset_manifest.csv`.
  - Storage notice: Full downloads require ~3 GB download bandwidth and ~7 GB uncompressed disk space.

---

## Automated Testing & Verification

The codebase includes two independent test suites verifying every component:

### 1. End-to-End Test Suite (`tests/test_end_to_end.py`)
```powershell
python -m pytest -v tests/test_end_to_end.py
```
**Test Coverage (8/8 Passed):**
1. `test_snn_forward_and_backward_cuda`: Tests GPU tensor execution, forward pass, and surrogate gradient backpropagation.
2. `test_neuron_state_reset`: Verifies that `functional.reset_net()` completely clears neuron membrane states.
3. `test_real_time_spike_monitor`: Validates probe hooks into all LIF layers and extraction of authentic firing metrics.
4. `test_benchmark_video_fixtures`: Checks video integrity, resolution (112x112), and frame rates of controlled fixtures.
5. `test_incident_neuron_hysteresis_and_temporal_filtering`: Tests Schmitt trigger state transitions (IDLE $\to$ ACTIVE $\to$ COOLDOWN $\to$ IDLE).
6. `test_evidence_clipping_standard_60s`: Verifies exact extraction of $[20.0\,\text{s}, 45.0\,\text{s}]$ for standard incident and SHA-256 verification.
7. `test_evidence_clipping_early_onset_boundary_clamp`: Verifies that $t_{\text{start}} - 10\,\text{s}$ clamps to $0.0\,\text{s}$ without crashing.
8. `test_dashboard_api_endpoints`: Verifies all REST endpoints (`/api/status`, `/api/evidence`, `/api/sample_videos`).

### 2. Standalone Smoke Tests (`test_smoke.py`)
```powershell
python test_smoke.py
```
**Stages Verified:**
- Config loading
- Single-step & multi-step SNN forward passes
- Baseline models (ResNet, MobileNet, MLP)
- Incident neuron hysteresis step
- ClipAssembler ring buffer & export
- Neuromorphic energy estimation
- Boundary evaluation metrics (tIoU, onset delay, offset error)
- Synthetic stream generator

---

## REST & WebSocket API Reference

| Endpoint | Method | Description |
| :--- | :--- | :--- |
| `/` | `GET` | Serves the main glassmorphic dashboard UI |
| `/api/status` | `GET` | Returns system telemetry, GPU name, VRAM allocation, and model status |
| `/api/evidence` | `GET` | Lists all finalized forensic clips with metadata and SHA-256 checksums |
| `/api/sample_videos` | `GET` | Lists available benchmark test streams for playback |
| `/api/start_stream` | `POST` | Starts the real-time SNN inference loop for a given video path |
| `/api/stop_stream` | `POST` | Stops active streaming inference and flushes the assembler ring buffer |
| `/api/upload` | `POST` | Accepts multipart video upload for inference |
| `/evidence/{clip_name}`| `GET` | Static video streaming endpoint for HTML5 `<video>` playback |
| `/ws/stream` | `WebSocket` | Real-time bidirectional telemetry stream delivering 16 FPS packets |

---

## License & Intellectual Property

- Code licensed under the **MIT License**.
- Developed as part of Machine Learning Project Assignment 8: *SNN-Based Real-Time Violence Detection, Neural Spike Visualization & Automated Evidence Dashboard*.
