import torch
import torch.nn.functional as F
from typing import Dict, Optional, Tuple, Union, Any

class IncidentNeuron:
    """
    IncidentNeuron (CORE NOVELTY):
    Leaky integrator with Schmitt trigger and hysteresis for temporal event boundary detection.
    """
    def __init__(self, beta: Union[float, Dict[str, Any]] = 0.9, th_on: float = 0.6, th_off: float = 0.3, min_on: int = 8, min_off: int = 16):
        if isinstance(beta, dict):
            cfg = beta
            self.beta = float(cfg.get("beta", 0.9))
            self.th_on = float(cfg.get("th_on", 0.6))
            self.th_off = float(cfg.get("th_off", 0.3))
            self.min_on = int(cfg.get("min_on", 8))
            self.min_off = int(cfg.get("min_off", 16))
        else:
            self.beta = float(beta)
            self.th_on = float(th_on)
            self.th_off = float(th_off)
            self.min_on = int(min_on)
            self.min_off = int(min_off)
        self.reset()
        
    def reset(self):
        """Resets the state to IDLE."""
        self.u = 0.0
        self.state = "IDLE"
        self.on_cnt = 0
        self.off_cnt = 0
        
    def step(self, evidence: float) -> Optional[str]:
        """
        Processes a single evidence frame and returns the event if any.
        evidence: float in [0, 1]
        """
        # Leaky integrate
        self.u = self.beta * self.u + (1.0 - self.beta) * evidence
        
        event = None
        
        if self.state == "IDLE":
            if self.u > self.th_on:
                self.on_cnt += 1
                if self.on_cnt >= self.min_on:
                    self.state = "ACTIVE"
                    event = "ONSET"
                    self.on_cnt = 0
            else:
                self.on_cnt = 0
                
        elif self.state == "ACTIVE":
            if self.u < self.th_off:
                self.off_cnt += 1
                if self.off_cnt >= self.min_off:
                    self.state = "IDLE"
                    event = "OFFSET"
                    self.off_cnt = 0
            else:
                self.off_cnt = 0
                
        return event

    def forward(self, evidence: float) -> Tuple[bool, Optional[str], float]:
        """
        Stream runner compatibility alias:
        Returns: (is_incident: bool, event: Optional[str], membrane_potential: float)
        """
        event = self.step(evidence)
        return (self.state == "ACTIVE", event, self.u)

    def get_state(self) -> Dict[str, Union[float, str, int]]:
        """Returns the current internal state."""
        return {
            "u": self.u,
            "state": self.state,
            "on_cnt": self.on_cnt,
            "off_cnt": self.off_cnt,
            "is_active": (self.state == "ACTIVE")
        }

    def process_sequence(self, evidence_list, fps: float = 16.0) -> list:
        """
        Processes a full sequence of evidence scores and extracts detected incident intervals.
        Returns list of intervals with start_seconds, end_seconds, start_frame, end_frame.
        """
        self.reset()
        incidents = []
        current_onset = None

        for idx, ev in enumerate(evidence_list):
            event = self.step(float(ev))
            if event == "ONSET":
                current_onset = idx
            elif event == "OFFSET" and current_onset is not None:
                incidents.append({
                    "start_frame": current_onset,
                    "end_frame": idx,
                    "start_seconds": round(current_onset / fps, 3),
                    "end_seconds": round(idx / fps, 3),
                    "duration_seconds": round((idx - current_onset) / fps, 3)
                })
                current_onset = None

        # If stream finishes while still active
        if self.state == "ACTIVE" and current_onset is not None:
            last_idx = len(evidence_list) - 1
            incidents.append({
                "start_frame": current_onset,
                "end_frame": last_idx,
                "start_seconds": round(current_onset / fps, 3),
                "end_seconds": round(last_idx / fps, 3),
                "duration_seconds": round((last_idx - current_onset) / fps, 3)
            })

        return incidents

class EvidenceExtractor:
    """
    Extracts evidence float in [0, 1] from raw model outputs.
    """
    def __init__(self, mode: str = 'softmax', window: int = 8):
        self.mode = mode
        self.window = window
        self.spike_history = []
        
    def __call__(self, spikes: torch.Tensor, membrane_logits: torch.Tensor, violence_class_idx: int = 1) -> float:
        """
        Extracts evidence.
        membrane_logits shape: [num_classes] or [B, num_classes] (assumed batch size 1 for inference)
        """
        if self.mode == 'softmax':
            # Use membrane_logits
            logits = membrane_logits
            if logits.dim() > 1:
                logits = logits.squeeze(0)
            probs = F.softmax(logits, dim=-1)
            evidence = probs[violence_class_idx].item()
            return evidence
            
        elif self.mode == 'spike_rate':
            # Use spikes
            if spikes.dim() > 1:
                spikes = spikes.squeeze(0)
            violence_spike = spikes[violence_class_idx].item()
            self.spike_history.append(violence_spike)
            if len(self.spike_history) > self.window:
                self.spike_history.pop(0)
            
            evidence = sum(self.spike_history) / len(self.spike_history)
            return evidence
            
        else:
            raise ValueError(f"Unknown mode: {self.mode}")

class BaselineThresholdDetector:
    """
    Simple sliding-window threshold detector for comparison.
    """
    def __init__(self, threshold: float = 0.5, window: int = 16):
        self.threshold = threshold
        self.window = window
        self.history = []
        
    def step(self, evidence: float) -> bool:
        """
        Returns True if sliding window average > threshold.
        evidence: float in [0, 1]
        """
        self.history.append(evidence)
        if len(self.history) > self.window:
            self.history.pop(0)
            
        avg = sum(self.history) / len(self.history)
        return avg > self.threshold


if __name__ == "__main__":
    # Test IncidentNeuron
    neuron = IncidentNeuron(beta=0.5, th_on=0.6, th_off=0.3, min_on=2, min_off=2)
    evidences = [0.1, 0.2, 0.8, 0.9, 0.9, 0.9, 0.2, 0.1, 0.1, 0.1]
    for i, ev in enumerate(evidences):
        event = neuron.step(ev)
        state = neuron.get_state()
        print(f"Step {i} | Evidence: {ev} | u: {state['u']:.2f} | State: {state['state']} | Event: {event}")
        
    # Test EvidenceExtractor
    extractor = EvidenceExtractor(mode='softmax')
    logits = torch.tensor([0.2, 1.5])
    spikes = torch.tensor([0.0, 1.0])
    ev = extractor(spikes, logits)
    print(f"Evidence (softmax): {ev:.4f}")
    
    extractor_spike = EvidenceExtractor(mode='spike_rate', window=3)
    for _ in range(5):
        ev = extractor_spike(spikes, logits)
    print(f"Evidence (spike_rate): {ev:.4f}")
    
    # Test BaselineThresholdDetector
    baseline = BaselineThresholdDetector(threshold=0.5, window=3)
    for ev in evidences:
        res = baseline.step(ev)
        print(f"Baseline detect: {res}")
