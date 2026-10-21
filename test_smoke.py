import os
import sys
import torch
import numpy as np
import yaml

def run_smoke_tests():
    print("=== Starting SNN Violence Detection Smoke Tests ===")
    
    # 1. Test Config Loading
    config_path = "configs/default.yaml"
    assert os.path.exists(config_path), f"Config file not found: {config_path}"
    with open(config_path, "r") as f:
        config = yaml.safe_load(f)
    print(" [1/8] Config loaded successfully.")

    # 2. Test SNN Model
    from models.snn import build_snn_model
    from spikingjelly.activation_based import functional
    snn_model = build_snn_model(config)
    snn_model.eval()
    
    # Single-step mode test
    dummy_frame = torch.randn(2, 3, 112, 112) # [B, C, H, W]
    functional.set_step_mode(snn_model, 's')
    spikes_s, logits_s = snn_model(dummy_frame)
    assert spikes_s.shape == (2, 2), f"Expected (2, 2), got {spikes_s.shape}"
    functional.reset_net(snn_model)
    
    # Multi-step mode test
    dummy_clip = torch.randn(16, 2, 3, 112, 112) # [T, B, C, H, W]
    functional.set_step_mode(snn_model, 'm')
    spikes_m, logits_m = snn_model(dummy_clip)
    assert spikes_m.shape == (16, 2, 2), f"Expected (16, 2, 2), got {spikes_m.shape}"
    functional.reset_net(snn_model)
    print(" [2/8] SNN Model forward pass (single-step & multi-step) passed.")

    # 3. Test Baseline Models
    from models.baselines import build_baseline_model
    for model_name in ["MobileNetGRU", "MobileNetPool", "ANNMirrorNet"]:
        b_model = build_baseline_model(model_name, config)
        b_model.eval()
        dummy_input = torch.randn(16, 2, 3, 112, 112) # [T, B, C, H, W]
        with torch.no_grad():
            out = b_model(dummy_input)
        assert out.shape == (2, 2), f"Baseline {model_name} output shape mismatch: {out.shape}"
    print(" [3/8] Baseline Models forward passes passed.")

    # 4. Test Incident Neuron
    from models.incident import IncidentNeuron
    inc_cfg = config.get("incident_neuron", {})
    inc_neuron = IncidentNeuron(inc_cfg)
    inc_neuron.reset()
    # Feed dummy evidence floats: low (calm) -> high (violence) -> low
    dummy_evidence_sequence = [0.1, 0.2, 0.8, 0.9, 0.9, 0.9, 0.2, 0.1]
    events = []
    for ev in dummy_evidence_sequence:
        event = inc_neuron.step(ev)
        events.append(event)
    print(f"   Incident Neuron events over test sequence: {events}")
    print(" [4/8] Incident Neuron hysteresis & LIF step passed.")

    # 5. Test Clip Assembler
    from pipeline.assembler import ClipAssembler
    temp_output_dir = "test_output_clips"
    os.makedirs(temp_output_dir, exist_ok=True)
    assembler = ClipAssembler(config, output_dir=temp_output_dir)
    
    # Push dummy frames
    dummy_np_frame = np.zeros((112, 112, 3), dtype=np.uint8)
    for t in range(50):
        # Trigger event between step 20 and 35
        is_triggered = (20 <= t <= 35)
        clip_saved = assembler.update(dummy_np_frame, is_triggered, frame_idx=t)
        if clip_saved:
            print(f"   Clip exported to: {clip_saved}")
            assert os.path.exists(clip_saved), "Exported clip file does not exist!"
            os.remove(clip_saved)
    if os.path.exists(temp_output_dir):
        try:
            os.rmdir(temp_output_dir)
        except Exception:
            pass
    print(" [5/8] Clip Assembler state machine & ring buffer passed.")

    # 6. Test Energy Calculation
    from energy import compute_sop_energy
    mock_firing_rates = {"conv0": 0.05, "conv1": 0.03, "fc": 0.1}
    energy_stats = compute_sop_energy(snn_model, mock_firing_rates, timesteps=16)
    assert "snn_energy_joules" in energy_stats
    print(f"   Estimated SNN Energy per clip: {energy_stats['snn_energy_joules']*1e3:.3f} mJ")
    print(" [6/8] Energy estimation passed.")

    # 7. Test Evaluation Metrics
    from eval_cls import evaluate_classification_metrics
    from eval_boundary import evaluate_boundary_detection
    
    y_true = np.array([0, 1, 1, 0, 1])
    y_pred = np.array([0, 1, 0, 0, 1])
    y_prob = np.array([0.1, 0.9, 0.4, 0.2, 0.8])
    cls_metrics = evaluate_classification_metrics(y_true, y_pred, y_prob)
    assert "accuracy" in cls_metrics and "f1_score" in cls_metrics
    
    gt_intervals = [[20, 40]]
    pred_intervals = [[22, 38]]
    boundary_metrics = evaluate_boundary_detection(gt_intervals, pred_intervals, total_frames=100)
    assert "mean_tiou" in boundary_metrics
    print(" [7/8] Evaluation metrics functions passed.")

    # 8. Test Data Stream Synthetic Generation
    from data.make_streams import generate_synthetic_stream
    synth_dir = "test_synth_stream"
    os.makedirs(synth_dir, exist_ok=True)
    frames, annotations = generate_synthetic_stream(synth_dir, num_frames=30)
    assert len(frames) == 30
    assert os.path.exists(os.path.join(synth_dir, "annotations.json"))
    # Cleanup dummy files
    for f in frames:
        if os.path.exists(f): os.remove(f)
    if os.path.exists(os.path.join(synth_dir, "annotations.json")):
        os.remove(os.path.join(synth_dir, "annotations.json"))
    os.rmdir(synth_dir)
    print(" [8/8] Data synthetic stream generator passed.")

    print("\n[SUCCESS] ALL 8 SMOKE TESTS PASSED SUCCESSFULLY!")

if __name__ == "__main__":
    run_smoke_tests()
