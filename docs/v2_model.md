# V2 Model Architecture: GenreCNN

## Overview
The **GenreCNN V2** model upgrades the baseline convolutional neural network by adding residual shortcuts (inspired by ResNet) and Squeeze-and-Excitation attention blocks (SE-Net). This enables deeper feature representations, speeds up gradient flow, and helps the model learn to focus on the frequency channels that are most representative for specific musical genres (e.g. low-frequency bass lines for Hip-Hop, high-frequency transients for Electronic).

---

## Architecture Breakdown

### 1. Squeeze-and-Excitation Block (`SEBlock`)
The Squeeze-and-Excitation attention mechanism models relationships between different feature channels to learn channel-wise importance weights.

```text
Input Tensor: (B, C, H, W)
      │
      ▼
[Global Avg Pool] (Squeeze) ───► shape: (B, C, 1, 1)
      │
      ▼
   [Flatten]                ───► shape: (B, C)
      │
      ▼
 [Linear (C -> C//16)]
      │
      ▼
    [ReLU]                  (Excitation Bottleneck)
      │
      ▼
 [Linear (C//16 -> C)]
      │
      ▼
   [Sigmoid]                ───► shape: (B, C)
      │
      ▼
   [Reshape]                ───► Channel Weights: (B, C, 1, 1)
      │
      ▼
[Multiply with Input]       ───► Rescaled Output: (B, C, H, W)
```

### 2. Residual Convolutional Block (`ResConvBlock`)
The core building block of the model integrates two standard convolutional layers, a skip connection, and a Squeeze-and-Excitation block.

- **Main Path**: `Conv2d (3x3) -> BatchNorm2d -> ReLU -> Conv2d (3x3) -> BatchNorm2d -> SEBlock`
- **Skip Connection**: 
  - If input channels match output channels: `nn.Identity()` (direct residual shortcut).
  - If input channels differ from output channels: A `1x1` Conv projection followed by `BatchNorm2d` to match dimensions before addition.
- **Combined Output**: `ReLU(Main Path + Skip Path) -> MaxPool2d (2x2)`

---

## Detailed Model Layers

| Layer / Block Name | Input Shape | Output Shape | Details |
| :--- | :--- | :--- | :--- |
| **Input Spectrogram** | `(B, 1, 128, T)` | `(B, 1, 128, T)` | Single-channel Log-Mel spectrogram |
| **Block 1 (ResConvBlock)** | `(B, 1, 128, T)` | `(B, 32, 64, T//2)` | 1 to 32 channels. Skip projection uses 1x1 Conv. |
| **Block 2 (ResConvBlock)** | `(B, 32, 64, T//2)` | `(B, 64, 32, T//4)` | 32 to 64 channels. Skip projection uses 1x1 Conv. |
| **Block 3 (ResConvBlock)** | `(B, 64, 32, T//4)` | `(B, 128, 16, T//8)` | 64 to 128 channels. Skip projection uses 1x1 Conv. |
| **Block 4 (ResConvBlock)** | `(B, 128, 16, T//8)` | `(B, 256, 8, T//16)` | 128 to 256 channels. Skip projection uses 1x1 Conv. |
| **Global Pool (GAP)** | `(B, 256, 8, T//16)` | `(B, 256, 1, 1)` | `AdaptiveAvgPool2d(1)` collapsing spatial dimensions |
| **Dense Head** | `(B, 256)` | `(B, 8)` | `Flatten -> Dropout(0.3) -> Linear(256->128) -> ReLU -> Dropout(0.15) -> Linear(128->8)` |

---

## Architectural Rationale

1. **Residual Connections (Skip Paths)**: By allowing gradients to flow directly through skip connections, we mitigate the vanishing gradient problem. This ensures stable optimization even during longer training sessions.
2. **Squeeze-and-Excitation Attention**: Standard CNNs treat all channels equally in each layer. SE blocks learn to emphasize useful feature maps (e.g. key rhythmic frequencies) while suppressing irrelevant ones, leading to an F1-score improvement across almost all genres.
3. **Global Average Pooling (GAP)**: Replacing massive dense flat layers with a simple Average Pooling layer reduces the parameter footprint by over 60%, heavily mitigating overfitting and making the model fully adaptive to variable-length audio input sequences.
