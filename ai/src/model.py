import torch
import torch.nn as nn


class SEBlock(nn.Module):
    """Squeeze-and-Excitation attention block.
    
    Learns channel-wise importance weights — which frequency bands
    matter most for a given genre (e.g. low frequencies for Hip-Hop,
    high frequencies for Electronic).
    
    Args:
        channels: Number of input/output channels.
        reduction: Bottleneck reduction ratio for the FC layers.
    """
    def __init__(self, channels: int, reduction: int = 16):
        super().__init__()
        self.squeeze = nn.AdaptiveAvgPool2d(1)
        self.excitation = nn.Sequential(
            nn.Flatten(),
            nn.Linear(channels, channels // reduction, bias=False),
            nn.ReLU(inplace=True),
            nn.Linear(channels // reduction, channels, bias=False),
            nn.Sigmoid(),
        )

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        b, c, _, _ = x.shape
        scale = self.squeeze(x)               # (B, C, 1, 1)
        scale = self.excitation(scale)        # (B, C)
        scale = scale.view(b, c, 1, 1)       # (B, C, 1, 1)
        return x * scale                      # channel-wise rescaling


class ResConvBlock(nn.Module):
    """Residual convolutional block with SE attention.
    
    Structure:
        x → Conv → BN → ReLU → Conv → BN → SE → + skip → ReLU → MaxPool
        
    The skip connection uses a 1×1 conv projection when in/out channels differ,
    ensuring gradient flow regardless of depth.
    
    Args:
        in_channels: Number of input channels.
        out_channels: Number of output channels.
        pool: Whether to apply MaxPool2d after the block.
        se_reduction: SE block bottleneck ratio.
    """
    def __init__(self, in_channels: int, out_channels: int, pool: bool = True, se_reduction: int = 16):
        super().__init__()

        self.conv_path = nn.Sequential(
            nn.Conv2d(in_channels, out_channels, kernel_size=3, padding=1, bias=False),
            nn.BatchNorm2d(out_channels),
            nn.ReLU(inplace=True),
            nn.Conv2d(out_channels, out_channels, kernel_size=3, padding=1, bias=False),
            nn.BatchNorm2d(out_channels),
        )

        self.se = SEBlock(out_channels, reduction=se_reduction)

        # 1×1 projection for skip connection when channels change
        self.skip = nn.Sequential(
            nn.Conv2d(in_channels, out_channels, kernel_size=1, bias=False),
            nn.BatchNorm2d(out_channels),
        ) if in_channels != out_channels else nn.Identity()

        self.relu = nn.ReLU(inplace=True)
        self.pool = nn.MaxPool2d(2, 2) if pool else nn.Identity()

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        residual = self.skip(x)               # match channels for addition
        out = self.conv_path(x)               # main conv path
        out = self.se(out)                    # channel attention
        out = self.relu(out + residual)       # residual addition
        return self.pool(out)


class GenreCNN(nn.Module):
    """CNN-based music genre classifier with residual connections and SE attention.
    
    Treats mel-spectrograms as 2D images. Four residual blocks progressively
    extract local (rhythm, beat) and global (timbre, harmony) features.
    Global Average Pooling collapses spatial dimensions, making the model
    robust to variable-length audio inputs.
    
    Args:
        num_classes: Number of output genre classes.
        dropout: Dropout rate for the classifier head.
        
    Input shape:  (B, 1, 128, T)  — single-channel mel-spectrogram
    Output shape: (B, num_classes) — raw logits
    """
    def __init__(self, num_classes: int = 8, dropout: float = 0.3):
        super().__init__()

        self.features = nn.Sequential(
            ResConvBlock(1,   32,  pool=True),    # → (B, 32,  64, T/2)
            ResConvBlock(32,  64,  pool=True),    # → (B, 64,  32, T/4)
            ResConvBlock(64,  128, pool=True),    # → (B, 128, 16, T/8)
            ResConvBlock(128, 256, pool=True),    # → (B, 256,  8, T/16)
        )

        # Global Average Pooling — collapses (H, W) → (1, 1)
        self.gap = nn.AdaptiveAvgPool2d(1)

        self.classifier = nn.Sequential(
            nn.Flatten(),
            nn.Dropout(dropout),
            nn.Linear(256, 128),
            nn.ReLU(inplace=True),
            nn.Dropout(dropout / 2),
            nn.Linear(128, num_classes),
        )

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        x = self.features(x)
        x = self.gap(x)
        return self.classifier(x)


def build_model(cfg: dict, device: torch.device) -> GenreCNN:
    """Instantiate and move GenreCNN to the target device.
    
    Args:
        cfg: Loaded config dict (from config.yaml).
        device: torch.device to move the model to.
        
    Returns:
        GenreCNN model on the specified device.
    """
    model = GenreCNN(
        num_classes=cfg["model"]["num_classes"],
        dropout=cfg["model"]["dropout"],
    )
    return model.to(device)


if __name__ == "__main__":
    model = GenreCNN()
    x = torch.randn(4, 1, 128, 1292)
    out = model(x)
    print(f"Input:  {x.shape}")
    print(f"Output: {out.shape}")
    params = sum(p.numel() for p in model.parameters() if p.requires_grad)
    print(f"Trainable params: {params:,}")