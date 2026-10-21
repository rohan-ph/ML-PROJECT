"""
VisionGuard SNN - Video Stream & File Inference Pipeline
=========================================================
Runs real-time temporal inference over video files or camera streams:
  - Frame difference calculation x_t = |I_t - I_{t-1}|
  - Single-step SNN forward pass preserving membrane states
  - Real-time spike monitoring and telemetry
  - IncidentNeuron state tracking & Schmitt trigger hysteresis
  - Automated 10s pre-event / 10s post-event evidence clip generation
  - Event streaming callbacks for real-time WebSocket dashboard
"""

import os
import cv2
import time
import argparse
import yaml
import torch
import numpy as np
from torchvision import transforms
from typing import Optional, Callable, Dict, Any, List

from models.snn import build_snn_model
from models.incident import IncidentNeuron, EvidenceExtractor
from pipeline.assembler import ClipAssembler
from pipeline.spike_monitor import RealTimeSpikeMonitor
from spikingjelly.activation_based import functional

class StreamRunner:
    def __init__(
        self,
        model: torch.nn.Module,
        incident_neuron: IncidentNeuron,
        clip_assembler: ClipAssembler,
        fps: int = 16,
        device: str = "cuda" if torch.cuda.is_available() else "cpu",
        on_telemetry_callback: Optional[Callable[[Dict[str, Any]], None]] = None
    ):
        self.model = model
        self.incident_neuron = incident_neuron
        self.clip_assembler = clip_assembler
        self.fps = fps
        self.device = torch.device(device)
        self.on_telemetry_callback = on_telemetry_callback

        self.model.to(self.device)
        self.model.eval()

        self.spike_monitor = RealTimeSpikeMonitor(self.model)

        self.transform = transforms.Compose([
            transforms.ToPILImage(),
            transforms.Resize((112, 112)),
            transforms.ToTensor()  # Scales to [0.0, 1.0] without negative distortion
        ])

    def _preprocess_frame(self, frame_bgr: np.ndarray, prev_frame_bgr: np.ndarray) -> torch.Tensor:
        """Computes absolute frame difference and converts to tensor [1, 3, 112, 112]."""
        diff = cv2.absdiff(frame_bgr, prev_frame_bgr)
        # Convert BGR to RGB
        diff_rgb = cv2.cvtColor(diff, cv2.COLOR_BGR2RGB)
        tensor = self.transform(diff_rgb).unsqueeze(0).to(self.device)
        return tensor

    def run(
        self,
        source: Any,
        visualize: bool = False,
        max_frames: Optional[int] = None
    ) -> List[Dict[str, Any]]:
        """
        Executes sequential inference over video source (file path or camera index).
        Returns list of captured telemetry events.
        """
        if isinstance(source, str) and source.isdigit():
            source = int(source)

        cap = cv2.VideoCapture(source)
        if not cap.isOpened():
            print(f"[Error] Failed to open video source: {source}")
            return []

        # Setup SNN in single-step streaming mode
        functional.set_step_mode(self.model, 's')
        functional.reset_net(self.model)
        self.incident_neuron.reset()
        self.spike_monitor.reset()

        ret, prev_frame = cap.read()
        if not ret:
            print("[Error] Empty video stream.")
            cap.release()
            return []

        frame_idx = 0
        all_telemetry = []
        source_fps = cap.get(cv2.CAP_PROP_FPS) or self.fps

        while True:
            if max_frames and frame_idx >= max_frames:
                break

            ret, frame = cap.read()
            if not ret:
                break

            # Compute consecutive frame difference
            input_tensor = self._preprocess_frame(frame, prev_frame)
            prev_frame = frame.copy()

            # Real SNN single-step forward pass
            with torch.no_grad():
                spikes, membrane_logits = self.model(input_tensor)

            # Calculate calibrated violence confidence
            if membrane_logits.shape[-1] >= 2:
                probs = torch.softmax(membrane_logits, dim=-1)
                violence_score = float(probs[0, 1].item())
            else:
                violence_score = float(torch.sigmoid(membrane_logits).item())

            pred_class = 1 if violence_score >= 0.5 else 0

            # Step IncidentNeuron Schmitt trigger
            event = self.incident_neuron.step(violence_score)
            neuron_state = self.incident_neuron.get_state()
            is_incident = (neuron_state["state"] == "ACTIVE")

            # Capture authentic neural spike telemetry
            timestamp_sec = frame_idx / self.fps
            telemetry = self.spike_monitor.capture_step(
                frame_idx=frame_idx,
                timestamp_sec=timestamp_sec,
                violence_score=violence_score,
                predicted_class=pred_class,
                incident_state=neuron_state["state"],
                incident_u=neuron_state["u"]
            )
            telemetry["event"] = event
            all_telemetry.append(telemetry)

            if self.on_telemetry_callback:
                self.on_telemetry_callback(telemetry)

            # Update evidence clip assembler
            self.clip_assembler.push(frame, event, is_incident, score=violence_score)

            # Optional OpenCV preview
            if visualize:
                vis = frame.copy()
                color = (0, 0, 255) if is_incident else (0, 255, 0)
                cv2.putText(vis, f"Status: {neuron_state['state']}", (10, 30),
                            cv2.FONT_HERSHEY_SIMPLEX, 0.7, color, 2)
                cv2.putText(vis, f"Violence Score: {violence_score:.3f}", (10, 60),
                            cv2.FONT_HERSHEY_SIMPLEX, 0.6, (255, 255, 255), 2)
                cv2.putText(vis, f"Neuron U: {neuron_state['u']:.3f}", (10, 90),
                            cv2.FONT_HERSHEY_SIMPLEX, 0.6, (0, 255, 255), 2)
                cv2.imshow("VisionGuard SNN Stream", vis)
                if cv2.waitKey(1) & 0xFF == ord('q'):
                    break

            frame_idx += 1

        cap.release()
        if visualize:
            cv2.destroyAllWindows()

        # Flush any remaining clip in active or cooldown state
        self.clip_assembler.flush()
        functional.reset_net(self.model)

        return all_telemetry
