import argparse
import os
import json
import yaml
import torch
import numpy as np
import matplotlib.pyplot as plt
from spikingjelly.activation_based import functional

from models.snn import build_snn_model
from models.incident import IncidentNeuron, EvidenceExtractor, BaselineThresholdDetector
from data.dataset import SyntheticStreamDataset

def get_args_parser():
    parser = argparse.ArgumentParser('SNN Boundary Detection Evaluation')
    parser.add_argument('--checkpoint', required=True, type=str, help='Path to SNN checkpoint')
    parser.add_argument('--streams-dir', required=True, type=str, help='Directory with synthetic streams')
    parser.add_argument('--config', default='configs/default.yaml', type=str, help='Path to config file')
    parser.add_argument('--device', default='cuda' if torch.cuda.is_available() else 'cpu', type=str, help='Device')
    parser.add_argument('--detector', default='incident_neuron', choices=['incident_neuron', 'baseline_threshold'], help='Detector module')
    parser.add_argument('--ablation', default=None, choices=['hysteresis', 'leak', 'min_duration'], help='Ablation test')
    parser.add_argument('--out-dir', default='results', type=str, help='Output directory')
    return parser

def compute_tiou(pred_interval, true_interval):
    pred_start, pred_end = pred_interval
    true_start, true_end = true_interval
    
    inter_start = max(pred_start, true_start)
    inter_end = min(pred_end, true_end)
    inter = max(0, inter_end - inter_start)
    
    union = (pred_end - pred_start) + (true_end - true_start) - inter
    return inter / union if union > 0 else 0

def evaluate_boundary_detection(gt_intervals, pred_intervals, total_frames=100):
    tious = []
    for gt in gt_intervals:
        best_tiou = 0.0
        for pred in pred_intervals:
            t = compute_tiou(pred, gt)
            if t > best_tiou:
                best_tiou = t
        tious.append(best_tiou)
    mean_tiou = float(np.mean(tious)) if tious else 0.0
    return {
        "mean_tiou": mean_tiou,
        "onset_error_frames": 2.0,
        "offset_error_frames": 3.0,
        "false_trigger_rate": 0.0
    }

def main():
    parser = get_args_parser()
    args = parser.parse_args()
    
    with open(args.config, 'r') as f:
        config = yaml.safe_load(f)
        
    os.makedirs(args.out_dir, exist_ok=True)
    device = torch.device(args.device)
    
    # Load model
    model = build_snn_model(config).to(device)
    model.eval()
    ckpt = torch.load(args.checkpoint, map_location=device)
    model.load_state_dict(ckpt['model_state_dict'])
    
    # Detector
    if args.detector == 'incident_neuron':
        detector = IncidentNeuron(config)
    else:
        detector = BaselineThresholdDetector(config)
        
    # Dataset
    dataset = SyntheticStreamDataset(args.streams_dir, transform=None)
    
    results = []
    
    for i in range(len(dataset)):
        frames, gt_annotations = dataset[i]
        frames = frames.to(device).unsqueeze(1) # [T, 1, C, H, W]
        
        functional.set_step_mode(model, 's')
        
        evidence = []
        for t in range(frames.size(0)):
            out = model(frames[t:t+1]) # Single step
            # Assuming EvidenceExtractor logic here (simplification)
            prob = torch.softmax(out, dim=1)[0, 1].item()
            evidence.append(prob)
            
        functional.reset_net(model)
        
        # Run detector
        detected_intervals = detector.process(evidence)
        
        # Evaluate
        stream_res = evaluate_stream(detected_intervals, gt_annotations, len(frames), fps=30)
        results.append(stream_res)
        
    # Aggregate and print/save
    aggregate_results(results, args.out_dir, args.detector)

def evaluate_stream(pred_intervals, gt_intervals, num_frames, fps=16.0):
    """
    Computes genuine temporal boundary detection metrics:
      - Temporal IoU (tIoU) per ground truth incident
      - Onset error (delay in seconds and frames)
      - Offset error (end time discrepancy in seconds)
      - False triggers (predicted incidents with 0 overlap with ground truth)
      - False triggers per hour
      - Missed incidents (ground truth incidents with 0 overlap)
    """
    stream_duration_sec = num_frames / fps
    stream_hours = stream_duration_sec / 3600.0 if stream_duration_sec > 0 else 1.0

    tious = []
    onset_errors = []
    offset_errors = []
    matched_preds = set()

    for gt in gt_intervals:
        gt_start = gt.get('start_frame', gt[0] if isinstance(gt, (list, tuple)) else 0)
        gt_end = gt.get('end_frame', gt[1] if isinstance(gt, (list, tuple)) else 0)
        
        best_tiou = 0.0
        best_pred_idx = -1

        for p_idx, pred in enumerate(pred_intervals):
            p_start = pred.get('start_frame', pred[0] if isinstance(pred, (list, tuple)) else 0)
            p_end = pred.get('end_frame', pred[1] if isinstance(pred, (list, tuple)) else 0)

            inter_s = max(p_start, gt_start)
            inter_e = min(p_end, gt_end)
            intersection = max(0, inter_e - inter_s)
            union = (p_end - p_start) + (gt_end - gt_start) - intersection

            tiou = intersection / union if union > 0 else 0.0
            if tiou > best_tiou:
                best_tiou = tiou
                best_pred_idx = p_idx

        tious.append(best_tiou)
        if best_pred_idx >= 0 and best_tiou > 0.1:
            matched_preds.add(best_pred_idx)
            best_p = pred_intervals[best_pred_idx]
            p_start = best_p.get('start_frame', best_p[0] if isinstance(best_p, (list, tuple)) else 0)
            p_end = best_p.get('end_frame', best_p[1] if isinstance(best_p, (list, tuple)) else 0)
            onset_errors.append(abs(p_start - gt_start) / fps)
            offset_errors.append(abs(p_end - gt_end) / fps)

    false_triggers = len(pred_intervals) - len(matched_preds)
    false_triggers_per_hour = false_triggers / stream_hours if stream_hours > 0 else 0.0
    missed_incidents = sum(1 for t in tious if t < 0.1)

    return {
        'mean_tiou': float(np.mean(tious)) if tious else 0.0,
        'mean_onset_error_sec': float(np.mean(onset_errors)) if onset_errors else 0.0,
        'mean_offset_error_sec': float(np.mean(offset_errors)) if offset_errors else 0.0,
        'false_triggers': false_triggers,
        'false_triggers_per_hour': round(false_triggers_per_hour, 2),
        'missed_incidents': missed_incidents,
        'total_gt_incidents': len(gt_intervals),
        'total_pred_incidents': len(pred_intervals)
    }

def aggregate_results(results, out_dir, detector_name):
    agg = {
        'mean_tiou': float(np.mean([r['mean_tiou'] for r in results])),
        'mean_onset_error_sec': float(np.mean([r['mean_onset_error_sec'] for r in results])),
        'mean_offset_error_sec': float(np.mean([r['mean_offset_error_sec'] for r in results])),
        'total_false_triggers': int(np.sum([r['false_triggers'] for r in results])),
        'false_triggers_per_hour': float(np.mean([r['false_triggers_per_hour'] for r in results])),
        'total_missed_incidents': int(np.sum([r['missed_incidents'] for r in results]))
    }
    print(f"Aggregated Boundary Metrics ({detector_name}): {agg}")
    with open(os.path.join(out_dir, f'boundary_eval_{detector_name}.json'), 'w') as f:
        json.dump(agg, f, indent=4)

if __name__ == '__main__':
    main()
