import torch
import torch.nn as nn
from spikingjelly.activation_based import neuron, layer, surrogate, functional
from typing import Tuple, Dict, Any, Optional

class SpikingViolenceNet(nn.Module):
    """
    Spiking Neural Network for Violence Detection.
    Architecture: 4x (Conv3x3 -> BN -> LIF -> MaxPool2x2) -> GAP -> FC -> LIF readout
    """
    def __init__(self, config: Optional[Dict[str, Any]] = None):
        super().__init__()
        if config is None:
            config = {}
        
        c_in = 3
        channels = [32, 64, 128, 256]
        num_classes = config.get("num_classes", 2)
        tau_conv = config.get("tau_conv", 2.0)
        tau_readout = config.get("tau_readout", 4.0)
        
        self.conv_blocks = nn.Sequential()
        in_ch = c_in
        for i, out_ch in enumerate(channels):
            self.conv_blocks.add_module(f"conv{i}", layer.Conv2d(in_ch, out_ch, kernel_size=3, padding=1, bias=False))
            self.conv_blocks.add_module(f"bn{i}", layer.BatchNorm2d(out_ch))
            self.conv_blocks.add_module(
                f"lif{i}", 
                neuron.LIFNode(
                    tau=tau_conv, 
                    surrogate_function=surrogate.ATan(alpha=2.0), 
                    detach_reset=True, 
                    step_mode='s'
                )
            )
            self.conv_blocks.add_module(f"pool{i}", layer.MaxPool2d(2, 2))
            in_ch = out_ch
            
        self.gap = layer.AdaptiveAvgPool2d((1, 1))
        self.flatten = layer.Flatten()
        
        self.fc = layer.Linear(channels[-1], num_classes)
        self.readout_lif = neuron.LIFNode(
            tau=tau_readout, 
            surrogate_function=surrogate.ATan(alpha=2.0), 
            detach_reset=True, 
            step_mode='s'
        )
        
        self.firing_rates: Dict[str, float] = {}
        self._register_hooks()
        
    def _register_hooks(self):
        self._hook_handles = []
        def hook_fn(name):
            def hook(module, input, output):
                if hasattr(output, "detach"):
                    spikes = output.detach()
                    # Calculate rate: sum of spikes / total number of elements
                    rate = spikes.mean().item()
                    self.firing_rates[name] = rate
            return hook
            
        for name, module in self.named_modules():
            if isinstance(module, neuron.LIFNode):
                handle = module.register_forward_hook(hook_fn(name))
                self._hook_handles.append(handle)
                
    def get_firing_rates(self) -> Dict[str, float]:
        """Returns a dictionary of per-layer firing rates."""
        return self.firing_rates
        
    def forward(self, x: torch.Tensor) -> Tuple[torch.Tensor, torch.Tensor]:
        """
        Forward pass for a single timestep or multiple timesteps based on step_mode.
        x shape: [B, C, H, W] for single-step, [T, B, C, H, W] for multi-step.
        """
        features = self.conv_blocks(x)
        features = self.gap(features)
        features = self.flatten(features)
        membrane_logits = self.fc(features)
        spikes = self.readout_lif(membrane_logits)
        
        return spikes, membrane_logits

def build_snn_model(config: Dict[str, Any]) -> SpikingViolenceNet:
    """Builds the SNN model from a configuration dictionary."""
    if "model" in config and "snn" in config["model"]:
        model_cfg = config["model"]["snn"]
    else:
        model_cfg = config
    return SpikingViolenceNet(model_cfg)

if __name__ == "__main__":
    # Test block
    config = {
        "model": {
            "snn": {
                "num_classes": 2,
                "tau_conv": 2.0,
                "tau_readout": 4.0
            }
        }
    }
    model = build_snn_model(config)
    
    # Test single-step mode
    x = torch.randn(4, 3, 112, 112)
    functional.set_step_mode(model, 's')
    spikes, logits = model(x)
    print(f"Single-step spikes shape: {spikes.shape}")
    print(f"Single-step logits shape: {logits.shape}")
    print(f"Firing rates: {model.get_firing_rates()}")
    
    # Reset network before switching mode or running new inputs
    functional.reset_net(model)
    
    # Test multi-step mode
    T = 8
    x_multi = torch.randn(T, 4, 3, 112, 112)
    functional.set_step_mode(model, 'm')
    spikes_m, logits_m = model(x_multi)
    print(f"Multi-step spikes shape: {spikes_m.shape}")
    print(f"Multi-step logits shape: {logits_m.shape}")
    print(f"Firing rates: {model.get_firing_rates()}")
