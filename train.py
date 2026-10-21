import argparse
import os
import yaml
import time
import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.tensorboard import SummaryWriter
from spikingjelly.activation_based import functional, monitor

# Assuming these modules exist based on the instructions
from models.snn import build_snn_model
from models.baselines import build_baseline_model
from data.dataset import get_dataloaders

def get_args_parser():
    parser = argparse.ArgumentParser('SNN Violence Detection Training', add_help=False)
    parser.add_argument('--model', default='snn', type=str, choices=['snn', 'mobilenet_gru', 'mobilenet_pool', 'ann_mirror'], help='Model type')
    parser.add_argument('--config', default='configs/default.yaml', type=str, help='Path to config file')
    parser.add_argument('--seed', default=None, type=int, help='Random seed')
    parser.add_argument('--resume', default='', type=str, help='Resume from checkpoint')
    parser.add_argument('--epochs', default=None, type=int, help='Number of epochs')
    parser.add_argument('--batch-size', default=None, type=int, help='Batch size')
    parser.add_argument('--lr', default=None, type=float, help='Learning rate')
    parser.add_argument('--device', default='cuda' if torch.cuda.is_available() else 'cpu', type=str, help='Device')
    parser.add_argument('--save-dir', default='checkpoints', type=str, help='Directory to save checkpoints')
    return parser

def set_seed(seed):
    import random
    import numpy as np
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed(seed)
        torch.cuda.manual_seed_all(seed)
        torch.backends.cudnn.deterministic = True

def train_epoch(model, dataloader, criterion, optimizer, device, epoch, writer, config, is_snn=True):
    model.train()
    if is_snn:
        functional.set_step_mode(model, 'm')
    
    total_loss = 0
    correct = 0
    total = 0
    
    spike_monitor = None
    if is_snn and config.get('training', {}).get('log_firing_rates', False):
        spike_monitor = monitor.OutputMonitor(model, neuron.LIFNode)
        
    start_time = time.time()
    for batch_idx, (inputs, targets) in enumerate(dataloader):
        inputs, targets = inputs.to(device), targets.to(device)
        
        # Ensure inputs are [T, B, C, H, W] for multi-step
        if len(inputs.shape) == 5 and not is_snn and not config.get('model', {}).get('needs_T', False):
            # Just an example adaptation if baselines don't take T
            pass
            
        optimizer.zero_grad()
        
        outputs = model(inputs)
        
        if is_snn:
            # SNN returns (spikes, membrane_logits), both shape [T, B, num_classes]
            spikes, membrane_logits = outputs
            out_mean = membrane_logits.mean(dim=0)  # Rate coding on membrane logits
            loss = criterion(out_mean, targets)
            _, predicted = out_mean.max(1)
            
            if spike_monitor:
                # Log firing rates
                for name, record in spike_monitor.records.items():
                    # mean over time and batch
                    firing_rate = record.mean().item()
                    writer.add_scalar(f'FiringRate/{name}', firing_rate, epoch * len(dataloader) + batch_idx)
                spike_monitor.clear_recorded_data()
        else:
            loss = criterion(outputs, targets)
            _, predicted = outputs.max(1)
            
        loss.backward()
        optimizer.step()
        
        if is_snn:
            functional.reset_net(model)
            
        total_loss += loss.item()
        total += targets.size(0)
        correct += predicted.eq(targets).sum().item()
        
    end_time = time.time()
    
    acc = 100. * correct / total
    avg_loss = total_loss / len(dataloader)
    
    return avg_loss, acc, end_time - start_time

def val_epoch(model, dataloader, criterion, device, is_snn=True):
    model.eval()
    if is_snn:
        functional.set_step_mode(model, 'm')
        
    total_loss = 0
    correct = 0
    total = 0
    
    with torch.no_grad():
        for inputs, targets in dataloader:
            inputs, targets = inputs.to(device), targets.to(device)
            
            outputs = model(inputs)
            
            if is_snn:
                spikes, membrane_logits = outputs
                out_mean = membrane_logits.mean(dim=0)
                loss = criterion(out_mean, targets)
                _, predicted = out_mean.max(1)
            else:
                loss = criterion(outputs, targets)
                _, predicted = outputs.max(1)
                
            if is_snn:
                functional.reset_net(model)
                
            total_loss += loss.item()
            total += targets.size(0)
            correct += predicted.eq(targets).sum().item()
            
    acc = 100. * correct / total
    avg_loss = total_loss / len(dataloader)
    
    return avg_loss, acc

def train_seed(args, config, seed):
    set_seed(seed)
    
    device = torch.device(args.device)
    os.makedirs(args.save_dir, exist_ok=True)
    
    # Override config with CLI
    epochs = args.epochs or config.get('training', {}).get('epochs', 50)
    batch_size = args.batch_size or config.get('training', {}).get('batch_size', 8)
    lr = args.lr or config.get('training', {}).get('lr', 1e-3)
    patience = config.get('training', {}).get('early_stopping_patience', 10)
    
    print(f"--- Training {args.model} with seed {seed} ---")
    
    # Model
    is_snn = (args.model == 'snn')
    if args.model == 'snn':
        model = build_snn_model(config).to(device)
    else:
        model = build_baseline_model(args.model, config).to(device)
        
    # Data
    train_loader, val_loader, _ = get_dataloaders(config, batch_size=batch_size)
    
    # Loss, Optimizer, Scheduler
    criterion = nn.CrossEntropyLoss()
    optimizer = optim.AdamW(model.parameters(), lr=lr)
    scheduler = optim.lr_scheduler.CosineAnnealingLR(optimizer, T_max=epochs)
    
    start_epoch = 0
    best_val_acc = 0.0
    best_val_loss = float('inf')
    epochs_no_improve = 0
    
    if args.resume:
        if os.path.isfile(args.resume):
            print(f"Loading checkpoint '{args.resume}'")
            checkpoint = torch.load(args.resume, map_location=device)
            start_epoch = checkpoint['epoch']
            best_val_acc = checkpoint['best_val_acc']
            best_val_loss = checkpoint['best_val_loss']
            model.load_state_dict(checkpoint['model_state_dict'])
            optimizer.load_state_dict(checkpoint['optimizer_state_dict'])
            scheduler.load_state_dict(checkpoint['scheduler_state_dict'])
            print(f"Loaded checkpoint at epoch {start_epoch}")
        else:
            print(f"No checkpoint found at '{args.resume}'")
            
    writer = SummaryWriter(log_dir=os.path.join('runs', f"{args.model}_seed{seed}"))
    
    for epoch in range(start_epoch, epochs):
        train_loss, train_acc, epoch_time = train_epoch(model, train_loader, criterion, optimizer, device, epoch, writer, config, is_snn=(args.model=='snn'))
        val_loss, val_acc = val_epoch(model, val_loader, criterion, device, is_snn=(args.model=='snn'))
        
        scheduler.step()
        
        # Logging
        writer.add_scalar('Loss/train', train_loss, epoch)
        writer.add_scalar('Loss/val', val_loss, epoch)
        writer.add_scalar('Accuracy/train', train_acc, epoch)
        writer.add_scalar('Accuracy/val', val_acc, epoch)
        writer.add_scalar('LR', scheduler.get_last_lr()[0], epoch)
        
        print(f"Epoch {epoch+1}/{epochs} - Time: {epoch_time:.2f}s - "
              f"Loss: {train_loss:.4f}, Acc: {train_acc:.2f}% - "
              f"Val Loss: {val_loss:.4f}, Val Acc: {val_acc:.2f}% - "
              f"LR: {scheduler.get_last_lr()[0]:.6f}")
              
        # Checkpointing
        checkpoint = {
            'epoch': epoch + 1,
            'model_state_dict': model.state_dict(),
            'optimizer_state_dict': optimizer.state_dict(),
            'scheduler_state_dict': scheduler.state_dict(),
            'best_val_acc': max(best_val_acc, val_acc),
            'best_val_loss': min(best_val_loss, val_loss),
            'config': config
        }
        torch.save(checkpoint, os.path.join(args.save_dir, f"{args.model}_{seed}_epoch{epoch+1}.pth"))
        
        if val_loss < best_val_loss:
            best_val_loss = val_loss
            epochs_no_improve = 0
        else:
            epochs_no_improve += 1
            
        if val_acc > best_val_acc:
            best_val_acc = val_acc
            torch.save(checkpoint, os.path.join(args.save_dir, f"{args.model}_{seed}_best.pth"))
            
        if epochs_no_improve >= patience:
            print(f"Early stopping triggered after {patience} epochs with no improvement in validation loss.")
            break
            
    writer.close()
    print(f"Training completed for seed {seed}. Best Val Acc: {best_val_acc:.2f}%")

def main():
    parser = get_args_parser()
    args = parser.parse_args()
    
    with open(args.config, 'r') as f:
        config = yaml.safe_load(f)
        
    seeds = config.get('training', {}).get('seeds', [42])
    if args.seed is not None:
        seeds = [args.seed]
        
    for seed in seeds:
        train_seed(args, config, seed)

if __name__ == '__main__':
    main()
