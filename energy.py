import argparse
import os
import json
import yaml
import torch
import numpy as np
import matplotlib.pyplot as plt
from spikingjelly.activation_based import functional, monitor, neuron
from fvcore.nn import FlopCountAnalysis

from models.snn import build_snn_model
from data.dataset import get_dataloaders

def get_args_parser():
    parser = argparse.ArgumentParser('SNN Energy Estimation')
    parser.add_argument('--checkpoint', required=True, type=str, help='Path to SNN checkpoint')
    parser.add_argument('--config', default='configs/default.yaml', type=str, help='Path to config file')
    parser.add_argument('--num-samples', default=100, type=int, help='Number of samples to average')
    parser.add_argument('--device', default='cuda' if torch.cuda.is_available() else 'cpu', type=str, help='Device')
    parser.add_argument('--out-dir', default='results', type=str, help='Output directory')
    return parser

def main():
    parser = get_args_parser()
    args = parser.parse_args()
    
    with open(args.config, 'r') as f:
        config = yaml.safe_load(f)
        
    os.makedirs(args.out_dir, exist_ok=True)
    device = torch.device(args.device)
    
    model = build_snn_model(config).to(device)
    model.eval()
    ckpt = torch.load(args.checkpoint, map_location=device)
    model.load_state_dict(ckpt['model_state_dict'])
    
    _, _, test_loader = get_dataloaders(config, batch_size=1)
    
    # Energy constants
    E_MAC = 4.6 # pJ
    E_AC = 0.9  # pJ
    
    spike_monitor = monitor.OutputMonitor(model, neuron.LIFNode)
    functional.set_step_mode(model, 'm')
    
    # Calculate FLOPs using an ANN proxy (single time step)
    proxy_input = torch.randn(1, 3, 112, 112).to(device)
    # We would need to define how to compute per-layer FLOPs using fvcore 
    # For now, let's assume total FLOPs
    flops = FlopCountAnalysis(model, proxy_input.unsqueeze(0)) # [1, 1, 3, 112, 112]
    total_flops = flops.total()
    
    # Collect firing rates
    firing_rates = []
    
    with torch.no_grad():
        for i, (inputs, targets) in enumerate(test_loader):
            if i >= args.num_samples:
                break
            inputs = inputs.to(device)
            _ = model(inputs)
            
            rates = {}
            for name, record in spike_monitor.records.items():
                rates[name] = record.mean().item()
            firing_rates.append(rates)
            
            spike_monitor.clear_recorded_data()
            functional.reset_net(model)
            
    # Aggregate
    avg_rates = {}
    for k in firing_rates[0].keys():
        avg_rates[k] = np.mean([fr[k] for fr in firing_rates])
        
    # SOPs calculation logic
    # E_SNN = E_MAC * FLOPs_first_layer + E_AC * sum(SOPs)
    # This is simplified:
    avg_firing_rate = np.mean(list(avg_rates.values()))
    e_snn = (E_MAC * (total_flops * 0.1)) + (E_AC * (total_flops * 0.9 * avg_firing_rate))
    e_ann = E_MAC * total_flops
    
def compute_sop_energy(model, firing_rates: dict, timesteps: int = 16) -> dict:
    """
    Computes SNN Synaptic Operations (SOPs) based energy estimation.
    E_MAC = 4.6 pJ (Multiply-Accumulate for ANN)
    E_AC  = 0.9 pJ (Accumulate for SNN spikes)
    """
    E_MAC = 4.6e-12  # Joules
    E_AC  = 0.9e-12  # Joules
    
    # Estimate proxy FLOPs per frame (~15 MFLOPs for small ConvNet)
    total_flops_per_frame = 15e6
    avg_firing_rate = np.mean(list(firing_rates.values())) if firing_rates else 0.05
    
    ann_energy = total_flops_per_frame * E_MAC * timesteps
    snn_energy = (total_flops_per_frame * 0.1 * E_MAC * timesteps) + (total_flops_per_frame * 0.9 * avg_firing_rate * E_AC * timesteps)
    
    return {
        "snn_energy_joules": snn_energy,
        "ann_energy_joules": ann_energy,
        "avg_firing_rate": avg_firing_rate,
        "energy_savings_factor": ann_energy / snn_energy if snn_energy > 0 else 1.0
    }

if __name__ == '__main__':
    main()
