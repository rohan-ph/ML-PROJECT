import torch
import torch.nn as nn
from torchvision.models import mobilenet_v2, MobileNet_V2_Weights
from typing import Dict, Any

class MobileNetGRU(nn.Module):
    """
    MobileNetV2 feature extractor followed by a GRU layer.
    """
    def __init__(self, num_classes: int = 2, hidden_size: int = 256, num_layers: int = 1, dropout: float = 0.3, pretrained: bool = True):
        super().__init__()
        
        weights = MobileNet_V2_Weights.IMAGENET1K_V1 if pretrained else None
        base_model = mobilenet_v2(weights=weights)
        self.feature_extractor = base_model.features
        self.pool = nn.AdaptiveAvgPool2d((1, 1))
        
        self.gru = nn.GRU(
            input_size=1280,
            hidden_size=hidden_size,
            num_layers=num_layers,
            batch_first=False, # Expects [T, B, C]
            dropout=dropout if num_layers > 1 else 0.0
        )
        self.dropout = nn.Dropout(dropout)
        self.fc = nn.Linear(hidden_size, num_classes)
        
    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """
        x shape: [T, B, 3, H, W]
        """
        T, B, C, H, W = x.shape
        # Merge T and B for feature extraction
        x_flat = x.view(T * B, C, H, W)
        features = self.feature_extractor(x_flat)
        features = self.pool(features)
        features = features.view(T, B, 1280)
        
        # Process through GRU
        output, hidden = self.gru(features)
        
        # Take the last hidden state for classification
        last_hidden = output[-1]
        out = self.dropout(last_hidden)
        out = self.fc(out)
        return out

class MobileNetPool(nn.Module):
    """
    MobileNetV2 feature extractor with temporal pooling.
    """
    def __init__(self, num_classes: int = 2, pretrained: bool = True):
        super().__init__()
        weights = MobileNet_V2_Weights.IMAGENET1K_V1 if pretrained else None
        base_model = mobilenet_v2(weights=weights)
        self.feature_extractor = base_model.features
        self.pool = nn.AdaptiveAvgPool2d((1, 1))
        self.fc = nn.Linear(1280, num_classes)
        
    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """
        x shape: [T, B, 3, H, W]
        """
        T, B, C, H, W = x.shape
        x_flat = x.view(T * B, C, H, W)
        features = self.feature_extractor(x_flat)
        features = self.pool(features)
        features = features.view(T, B, 1280)
        
        # Average over T
        avg_features = features.mean(dim=0)
        out = self.fc(avg_features)
        return out

class ANNMirrorNet(nn.Module):
    """
    Exact ANN equivalent of SpikingViolenceNet (uses ReLU instead of LIF).
    """
    def __init__(self, config: Dict[str, Any]):
        super().__init__()
        
        c_in = 3
        channels = [32, 64, 128, 256]
        if "model" in config and "ann" in config["model"]:
            model_cfg = config["model"]["ann"]
        else:
            model_cfg = config
        num_classes = model_cfg.get("num_classes", 2)
        
        self.conv_blocks = nn.Sequential()
        in_ch = c_in
        for i, out_ch in enumerate(channels):
            self.conv_blocks.add_module(f"conv{i}", nn.Conv2d(in_ch, out_ch, kernel_size=3, padding=1, bias=False))
            self.conv_blocks.add_module(f"bn{i}", nn.BatchNorm2d(out_ch))
            self.conv_blocks.add_module(f"relu{i}", nn.ReLU(inplace=True))
            self.conv_blocks.add_module(f"pool{i}", nn.MaxPool2d(2, 2))
            in_ch = out_ch
            
        self.gap = nn.AdaptiveAvgPool2d((1, 1))
        self.flatten = nn.Flatten()
        self.fc = nn.Linear(channels[-1], num_classes)
        
    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """
        x shape: [T, B, 3, H, W]
        Processes each frame, averages logits over T.
        """
        T, B, C, H, W = x.shape
        x_flat = x.view(T * B, C, H, W)
        
        features = self.conv_blocks(x_flat)
        features = self.gap(features)
        features = self.flatten(features)
        logits_flat = self.fc(features)
        
        logits = logits_flat.view(T, B, -1)
        # Average over T frames
        out = logits.mean(dim=0)
        return out

def build_baseline_model(name: str, config: Dict[str, Any]) -> nn.Module:
    """Builds the baseline model based on name and config."""
    if name == "MobileNetGRU":
        return MobileNetGRU(
            num_classes=config.get("num_classes", 2),
            hidden_size=config.get("hidden_size", 256),
            num_layers=config.get("num_layers", 1),
            dropout=config.get("dropout", 0.3),
            pretrained=config.get("pretrained", True)
        )
    elif name == "MobileNetPool":
        return MobileNetPool(
            num_classes=config.get("num_classes", 2),
            pretrained=config.get("pretrained", True)
        )
    elif name == "ANNMirrorNet":
        return ANNMirrorNet(config)
    else:
        raise ValueError(f"Unknown baseline model: {name}")

if __name__ == "__main__":
    # Test baselines
    T, B, C, H, W = 8, 2, 3, 112, 112
    x = torch.randn(T, B, C, H, W)
    
    gru_model = build_baseline_model("MobileNetGRU", {})
    gru_out = gru_model(x)
    print(f"MobileNetGRU output shape: {gru_out.shape}")
    
    pool_model = build_baseline_model("MobileNetPool", {})
    pool_out = pool_model(x)
    print(f"MobileNetPool output shape: {pool_out.shape}")
    
    ann_config = {"num_classes": 2}
    ann_model = build_baseline_model("ANNMirrorNet", ann_config)
    ann_out = ann_model(x)
    print(f"ANNMirrorNet output shape: {ann_out.shape}")
