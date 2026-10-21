import argparse
import os
import json
import yaml
import torch
import numpy as np
from sklearn.metrics import accuracy_score, precision_score, recall_score, f1_score, roc_auc_score, confusion_matrix, roc_curve
from scipy.stats import contingency
import matplotlib.pyplot as plt
from spikingjelly.activation_based import functional

from models.snn import build_snn_model
from models.baselines import build_baseline_model
from data.dataset import get_dataloaders

def get_args_parser():
    parser = argparse.ArgumentParser('SNN Classification Evaluation')
    parser.add_argument('--model', default='snn', type=str, help='Model type')
    parser.add_argument('--checkpoint', default=None, nargs='+', type=str, help='Path(s) to checkpoint(s)')
    parser.add_argument('--dataset', default='rwf2000', type=str, choices=['rwf2000', 'hockey', 'rlvs'], help='Dataset to evaluate on')
    parser.add_argument('--config', default='configs/default.yaml', type=str, help='Path to config file')
    parser.add_argument('--device', default='cuda' if torch.cuda.is_available() else 'cpu', type=str, help='Device')
    parser.add_argument('--out-dir', default='results', type=str, help='Output directory')
    return parser

def evaluate_classification_metrics(y_true, y_pred, y_prob):
    acc = accuracy_score(y_true, y_pred)
    prec = precision_score(y_true, y_pred, zero_division=0)
    rec = recall_score(y_true, y_pred, zero_division=0)
    f1 = f1_score(y_true, y_pred, zero_division=0)
    try:
        auc = roc_auc_score(y_true, y_prob)
    except Exception:
        auc = 0.5
    return {
        'accuracy': acc, 'precision': prec, 'recall': rec, 'f1_score': f1, 'auc_roc': auc
    }

def evaluate(model, dataloader, device, is_snn=True):
    model.eval()
    if is_snn:
        functional.set_step_mode(model, 'm')
        
    all_preds = []
    all_probs = []
    all_targets = []
    
    with torch.no_grad():
        for inputs, targets in dataloader:
            inputs = inputs.to(device)
            outputs = model(inputs)
            
            if is_snn:
                spikes, membrane_logits = outputs
                out_mean = membrane_logits.mean(dim=0)
                probs = torch.softmax(out_mean, dim=1)[:, 1].cpu().numpy()
                preds = out_mean.max(1)[1].cpu().numpy()
                functional.reset_net(model)
            else:
                probs = torch.softmax(outputs, dim=1)[:, 1].cpu().numpy()
                preds = outputs.max(1)[1].cpu().numpy()
                
            all_preds.extend(preds)
            all_probs.extend(probs)
            all_targets.extend(targets.numpy())
            
    return np.array(all_targets), np.array(all_preds), np.array(all_probs)

def main():
    parser = get_args_parser()
    args = parser.parse_args()
    
    with open(args.config, 'r') as f:
        config = yaml.safe_load(f)
        
    os.makedirs(args.out_dir, exist_ok=True)
    device = torch.device(args.device)
    
    # Modify config to point to the desired dataset for evaluation
    config['data']['dataset_name'] = args.dataset
    _, _, test_loader = get_dataloaders(config, batch_size=config.get('eval', {}).get('batch_size', 16))
    
    is_snn = (args.model == 'snn' or args.model == 'ann_mirror')
    
    metrics_list = []
    
    if not args.checkpoint:
        print("No checkpoint provided.")
        return
        
    for ckpt_path in args.checkpoint:
        if args.model == 'snn':
            model = build_snn_model(config).to(device)
        else:
            model = build_baseline_model(args.model, config).to(device)
            
        print(f"Loading checkpoint {ckpt_path}")
        checkpoint = torch.load(ckpt_path, map_location=device)
        model.load_state_dict(checkpoint['model_state_dict'])
        
        targets, preds, probs = evaluate(model, test_loader, device, is_snn=is_snn)
        
        acc = accuracy_score(targets, preds)
        prec = precision_score(targets, preds)
        rec = recall_score(targets, preds)
        f1 = f1_score(targets, preds)
        auc = roc_auc_score(targets, probs)
        
        cm = confusion_matrix(targets, preds)
        tn, fp, fn, tp = cm.ravel()
        fpr = fp / (fp + tn) if (fp + tn) > 0 else 0
        
        metrics = {
            'accuracy': acc, 'precision': prec, 'recall': rec, 'f1': f1,
            'auc_roc': auc, 'fpr': fpr, 'checkpoint': ckpt_path
        }
        metrics_list.append(metrics)
        
        # Plot ROC and CM for first checkpoint
        if len(metrics_list) == 1:
            fpr_curve, tpr_curve, _ = roc_curve(targets, probs)
            plt.figure()
            plt.plot(fpr_curve, tpr_curve, label=f'{args.model} (AUC = {auc:.2f})')
            plt.plot([0, 1], [0, 1], 'k--')
            plt.xlabel('False Positive Rate')
            plt.ylabel('True Positive Rate')
            plt.title('ROC Curve')
            plt.legend()
            plt.savefig(os.path.join(args.out_dir, f'roc_curve_{args.model}_{args.dataset}.png'))
            plt.close()
            
    # Aggregate if multiple checkpoints
    if len(metrics_list) > 1:
        agg = {}
        for k in ['accuracy', 'precision', 'recall', 'f1', 'auc_roc', 'fpr']:
            vals = [m[k] for m in metrics_list]
            agg[f'{k}_mean'] = np.mean(vals)
            agg[f'{k}_std'] = np.std(vals)
            print(f"{k}: {agg[f'{k}_mean']:.4f} ± {agg[f'{k}_std']:.4f}")
        
        out_file = os.path.join(args.out_dir, f'eval_{args.model}_{args.dataset}_multi.json')
        with open(out_file, 'w') as f:
            json.dump(agg, f, indent=4)
    else:
        for k, v in metrics_list[0].items():
            if k != 'checkpoint':
                print(f"{k}: {v:.4f}")
        out_file = os.path.join(args.out_dir, f'eval_{args.model}_{args.dataset}.json')
        with open(out_file, 'w') as f:
            json.dump(metrics_list[0], f, indent=4)

if __name__ == '__main__':
    main()
