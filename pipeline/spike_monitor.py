"""
VisionGuard SNN - Real-Time Spike Monitor & Instrumentation
============================================================
Captures genuine spiking neuron telemetry during inference:
  - Layer-wise spike counts and firing rates
  - Readout neuron membrane potentials (V_mem for Class 0 vs Class 1)
  - Probe neuron spikes for live raster plots
  - Total synaptic activity across the SNN backbone
"""

import torch
import torch.nn as nn
from typing import Dict, List, Any, Optional

class RealTimeSpikeMonitor:
    """
    Hooks into SpikingViolenceNet LIF nodes to capture authentic neural spike
    and membrane potential activity during frame-by-frame inference.
    """
    def __init__(self, model: nn.Module, num_probes: int = 16):
        self.model = model
        self.num_probes = num_probes
        self.handles = []
        self.current_frame_data: Dict[str, Any] = {}
        self.history: List[Dict[str, Any]] = []
        self._register_hooks()

    def _register_hooks(self):
        self.raw_layer_spikes: Dict[str, torch.Tensor] = {}
        self.raw_layer_vmem: Dict[str, torch.Tensor] = {}

        def get_hook(name):
            def hook(module, inp, output):
                # Output is spikes tensor [B, ...]
                if isinstance(output, torch.Tensor):
                    self.raw_layer_spikes[name] = output.detach()
                if hasattr(module, 'v') and isinstance(module.v, torch.Tensor):
                    self.raw_layer_vmem[name] = module.v.detach()
            return hook

        # Look for LIF nodes across model
        for name, module in self.model.named_modules():
            # Check class name or attribute to avoid hard import requirement
            if "LIFNode" in module.__class__.__name__ or hasattr(module, "v_threshold"):
                handle = module.register_forward_hook(get_hook(name))
                self.handles.append(handle)

    def capture_step(
        self,
        frame_idx: int,
        timestamp_sec: float,
        violence_score: float,
        predicted_class: int,
        incident_state: str,
        incident_u: float
    ) -> Dict[str, Any]:
        """
        Extracts metrics from the latest forward step and clears the raw buffers.
        """
        layer_firing_rates = {}
        layer_spike_counts = {}
        total_spikes = 0
        probe_spikes = []

        for name, spikes in self.raw_layer_spikes.items():
            count = int(spikes.sum().item())
            rate = float(spikes.float().mean().item())
            layer_firing_rates[name] = round(rate, 4)
            layer_spike_counts[name] = count
            total_spikes += count

            # Gather probe samples (up to 4 per layer)
            if len(probe_spikes) < self.num_probes:
                flat = spikes.view(-1)
                needed = min(4, self.num_probes - len(probe_spikes), len(flat))
                if needed > 0:
                    probe_spikes.extend([int(v.item()) for v in flat[:needed]])

        # Pad probe spikes if needed
        while len(probe_spikes) < self.num_probes:
            probe_spikes.append(0)

        # Extract readout membrane potential
        readout_v = [0.0, 0.0]
        for name, vmem in self.raw_layer_vmem.items():
            if "readout" in name or name.endswith("lif") or "fc" in name:
                flat_v = vmem.view(-1)
                if len(flat_v) >= 2:
                    readout_v = [round(float(flat_v[0].item()), 4), round(float(flat_v[1].item()), 4)]

        telemetry = {
            "frame_idx": int(frame_idx),
            "timestamp_seconds": round(float(timestamp_sec), 3),
            "violence_score": round(float(violence_score), 4),
            "predicted_class": int(predicted_class),
            "total_spikes": total_spikes,
            "layer_firing_rates": layer_firing_rates,
            "layer_spike_counts": layer_spike_counts,
            "readout_v_mem": readout_v,
            "probe_spikes": probe_spikes[:self.num_probes],
            "incident_state": str(incident_state),
            "incident_neuron_u": round(float(incident_u), 4)
        }

        self.current_frame_data = telemetry
        self.history.append(telemetry)

        # Clear raw cache for next step
        self.raw_layer_spikes.clear()
        self.raw_layer_vmem.clear()
        return telemetry

    def reset(self):
        """Resets the history buffer."""
        self.current_frame_data = {}
        self.history.clear()
        self.raw_layer_spikes.clear()
        self.raw_layer_vmem.clear()

    def remove_hooks(self):
        for h in self.handles:
            h.remove()
        self.handles.clear()
