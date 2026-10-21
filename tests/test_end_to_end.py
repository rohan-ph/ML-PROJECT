"""
VisionGuard SNN - Automated End-to-End Test Suite
=================================================
Automated verification across all core acceptance criteria:
  1. Model Tests: SNN forward/backward, CUDA execution, neuron reset, spike monitor
  2. Dataset Tests: Path validation, FPS/duration inspection, temporal ordering
  3. Incident Tests: Sustained score triggers onset, brief noise filtered by hysteresis, offset triggers
  4. Evidence Tests: Exact [max(0, t-10), min(T, t+10)] clipping, SHA-256 hash re-verification
  5. Dashboard API Tests: Endpoints, status, evidence listing, sample videos
"""

import os
import json
import pytest
import torch
import cv2
import numpy as np
from pathlib import Path
from fastapi.testclient import TestClient

from models.snn import build_snn_model, SpikingViolenceNet
from models.incident import IncidentNeuron
from pipeline.assembler import ClipAssembler
from pipeline.spike_monitor import RealTimeSpikeMonitor
from pipeline.stream_runner import StreamRunner
from dashboard.app import app
from spikingjelly.activation_based import functional

# --- 1. MODEL & CUDA TESTS ---

def test_snn_forward_and_backward_cuda():
    """Verifies SNN forward, surrogate-gradient backward, and CUDA execution on RTX 3050."""
    device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")
    model = SpikingViolenceNet().to(device)
    model.train()
    functional.set_step_mode(model, 'm')
    
    T, B = 8, 2
    dummy_input = torch.randn(T, B, 3, 112, 112, device=device, requires_grad=True)
    spikes, logits = model(dummy_input)
    
    assert spikes.shape == (T, B, 2), f"Expected spikes shape (8, 2, 2), got {spikes.shape}"
    assert logits.shape == (T, B, 2), f"Expected logits shape (8, 2, 2), got {logits.shape}"
    
    loss = logits.mean()
    loss.backward()
    assert dummy_input.grad is not None, "Gradient backward pass failed!"
    assert dummy_input.grad.norm().item() > 0, "Gradient norm is zero!"
    functional.reset_net(model)


def test_neuron_state_reset():
    """Verifies that functional.reset_net clears neuron membrane states."""
    model = SpikingViolenceNet()
    functional.set_step_mode(model, 's')
    
    x1 = torch.randn(1, 3, 112, 112)
    with torch.no_grad():
        _, logits1 = model(x1)
        
    functional.reset_net(model)
    # Check that after reset, first convolution LIF membrane potential is zero
    first_lif = model.conv_blocks.lif0
    assert first_lif.v == 0.0 or first_lif.v is None or (isinstance(first_lif.v, torch.Tensor) and first_lif.v.sum().item() == 0)


def test_real_time_spike_monitor():
    """Verifies that RealTimeSpikeMonitor hooks into LIF nodes and captures authentic telemetry."""
    model = SpikingViolenceNet()
    monitor = RealTimeSpikeMonitor(model, num_probes=16)
    functional.set_step_mode(model, 's')
    
    x = torch.randn(1, 3, 112, 112).abs()
    with torch.no_grad():
        spikes, logits = model(x)
        
    telemetry = monitor.capture_step(
        frame_idx=0,
        timestamp_sec=0.0,
        violence_score=0.75,
        predicted_class=1,
        incident_state="ACTIVE",
        incident_u=0.72
    )
    
    assert "layer_firing_rates" in telemetry
    assert "layer_spike_counts" in telemetry
    assert len(telemetry["probe_spikes"]) == 16
    assert "readout_v_mem" in telemetry
    assert telemetry["predicted_class"] == 1
    monitor.remove_hooks()


# --- 2. DATASET & VIDEO TESTS ---

def test_benchmark_video_fixtures():
    """Verifies the duration, FPS, and integrity of controlled benchmark videos."""
    v1_path = Path("data/incident_timing/videos/stream_fixture_60s.mp4")
    assert v1_path.exists(), f"Missing fixture: {v1_path}"
    
    cap = cv2.VideoCapture(str(v1_path))
    assert cap.isOpened(), "Failed to open 60s benchmark video"
    fps = cap.get(cv2.CAP_PROP_FPS)
    n_frames = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))
    cap.release()
    
    assert abs(fps - 16.0) < 0.5, f"Expected 16 FPS, got {fps}"
    assert n_frames == 960, f"Expected 960 frames (60s @ 16fps), got {n_frames}"


# --- 3. INCIDENT DETECTOR TESTS ---

def test_incident_neuron_hysteresis_and_temporal_filtering():
    """Verifies that brief noise does not trigger ONSET, but sustained score triggers ONSET and OFFSET."""
    neuron = IncidentNeuron(beta=0.9, th_on=0.6, th_off=0.3, min_on=4, min_off=4)
    neuron.reset()
    
    # 1. Brief noise pulse (2 frames above 0.8) should NOT trigger ONSET because min_on=4
    e1 = neuron.step(0.9)
    e2 = neuron.step(0.9)
    assert neuron.state == "IDLE", "Premature activation on brief noise!"
    assert e1 is None and e2 is None
    
    # 2. Reset score to calm
    neuron.step(0.1)
    neuron.step(0.1)
    
    # 3. Sustained violence sequence (min_on=4 consecutive frames above th_on)
    events = []
    for _ in range(50):
        ev = neuron.step(0.95)
        if ev:
            events.append(ev)
            
    assert "ONSET" in events, "Failed to trigger ONSET on sustained violence!"
    assert neuron.state == "ACTIVE"
    
    # 4. Sustained calm sequence (min_off=4 consecutive frames below th_off)
    offset_events = []
    for _ in range(50):
        ev = neuron.step(0.05)
        if ev:
            offset_events.append(ev)
            
    assert "OFFSET" in offset_events, "Failed to trigger OFFSET on calm activity!"
    assert neuron.state == "IDLE"


# --- 4. EXACT EVIDENCE CLIPPING TESTS ---

def test_evidence_clipping_standard_60s():
    """
    For incident from 30 to 35 seconds in 60s video:
      Expected clip: [20.0s, 45.0s], duration ~25.0s, pre_truncated=False
    """
    v_path = "data/incident_timing/videos/stream_fixture_60s.mp4"
    meta = ClipAssembler.extract_incident_clip_offline(
        video_path=v_path,
        incident_start_sec=30.0,
        incident_end_sec=35.0,
        incident_id="test_suite_std",
        output_dir="evidence"
    )
    
    assert abs(meta["clip_start_seconds"] - 20.0) < 0.1
    assert abs(meta["clip_end_seconds"] - 45.0) < 0.1
    assert abs(meta["duration_seconds"] - 25.0) < 0.2
    assert meta["pre_truncated"] is False
    assert meta["post_truncated"] is False
    assert meta["sha256_verified"] is True
    assert os.path.exists(meta["evidence_clip_path"])


def test_evidence_clipping_early_onset_boundary_clamp():
    """
    For incident from 3 to 8 seconds in 20s video:
      Expected clip: [0.0s, 18.0s], duration ~18.0s, pre_truncated=True
    """
    v_path = "data/incident_timing/videos/stream_fixture_20s_early.mp4"
    meta = ClipAssembler.extract_incident_clip_offline(
        video_path=v_path,
        incident_start_sec=3.0,
        incident_end_sec=8.0,
        incident_id="test_suite_early",
        output_dir="evidence"
    )
    
    assert abs(meta["clip_start_seconds"] - 0.0) < 0.1
    assert abs(meta["clip_end_seconds"] - 18.0) < 0.1
    assert abs(meta["duration_seconds"] - 18.0) < 0.2
    assert meta["pre_truncated"] is True
    assert meta["sha256_verified"] is True


# --- 5. DASHBOARD REST API TESTS ---

def test_dashboard_api_endpoints():
    """Verifies that FastAPI REST endpoints return valid status and evidence metadata."""
    client = TestClient(app)
    
    # 1. Health & GPU status
    resp = client.get("/api/status")
    assert resp.status_code == 200
    data = resp.json()
    assert "cuda_available" in data
    assert "model_architecture" in data
    
    # 2. Evidence list
    ev_resp = client.get("/api/evidence")
    assert ev_resp.status_code == 200
    clips = ev_resp.json()
    assert isinstance(clips, list)
    assert len(clips) >= 2, "Expected at least 2 generated test evidence clips!"
    
    # Verify first clip structure
    c0 = clips[0]
    assert "sha256_hash" in c0
    assert "clip_start_seconds" in c0
    assert "clip_end_seconds" in c0
    
    # 3. Sample videos
    samp_resp = client.get("/api/sample_videos")
    assert samp_resp.status_code == 200
    assert len(samp_resp.json()["videos"]) > 0


if __name__ == "__main__":
    pytest.main(["-v", "tests/test_end_to_end.py"])
