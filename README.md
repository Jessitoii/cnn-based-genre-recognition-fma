# CNN-Based Genre Recognition — FMA

> Automatic music genre classification using Convolutional Neural Networks on Mel-Spectrograms, trained on the Free Music Archive (FMA) dataset.

![Python](https://img.shields.io/badge/Python-3.10+-3776AB?style=flat&logo=python&logoColor=white)
![PyTorch](https://img.shields.io/badge/PyTorch-2.0+-EE4C2C?style=flat&logo=pytorch&logoColor=white)
![Librosa](https://img.shields.io/badge/Librosa-0.10+-green?style=flat)
![License](https://img.shields.io/badge/License-MIT-yellow?style=flat)
![Test Accuracy](https://img.shields.io/badge/Test%20Accuracy-60.63%25-blue?style=flat)

---

## Overview

This project demonstrates that CNN architectures — originally designed for image recognition — can effectively classify music genres by treating audio as visual data. Audio signals are transformed into **Mel-Spectrograms**, which encode time-frequency information as 2D images, and fed into a custom CNN classifier.

The core thesis: **a spectrogram is an image. If a CNN can distinguish a cat from a dog, it can distinguish Hip-Hop from Folk.**

---

### Usage (Hugging Face)

```python
from huggingface_hub import hf_hub_download
import torch

model_path = hf_hub_download(
    repo_id="jessitoi/genre-cnn-fma-small",
    filename="model.pth"
)

model = torch.load(model_path)
model.eval()
```

---

## Architecture

### Audio → Image Pipeline

```
MP3 File (30s clip)
       │
       ▼
  [Librosa Load]
  sr=22050Hz, mono
       │
       ▼
  [Mel-Spectrogram]
  n_mels=128, n_fft=2048
  hop_length=512
       │
       ▼
  [Power to dB]
  librosa.power_to_db()
       │
       ▼
  [Normalization]
  per-sample mean/std
       │
       ▼
  Tensor: (1, 128, 1292)
       │
       ▼
  [GenreCNN]
       │
       ▼
  Genre Prediction (8 classes)
```

### GenreCNN Architecture

```
Input: (B, 1, 128, T)
       │
       ▼
ConvBlock 1: Conv2d(1→32) → BN → ReLU → Conv2d(32→32) → BN → ReLU → MaxPool2d
ConvBlock 2: Conv2d(32→64) → BN → ReLU → Conv2d(64→64) → BN → ReLU → MaxPool2d
ConvBlock 3: Conv2d(64→128) → BN → ReLU → Conv2d(128→128) → BN → ReLU → MaxPool2d
ConvBlock 4: Conv2d(128→256) → BN → ReLU → Conv2d(256→256) → BN → ReLU → MaxPool2d
       │
       ▼
AdaptiveAvgPool2d(1)   ← Global Average Pooling
       │
       ▼
Flatten → Dropout(0.3) → Linear(256→128) → ReLU → Dropout(0.15) → Linear(128→8)
       │
       ▼
Output: (B, 8) logits
```

**Design decisions:**
- **Double Conv per block** — richer feature extraction before pooling, inspired by VGG
- **Global Average Pooling** — eliminates fully connected spatial layers, robust to input length variation, reduces overfitting
- **BatchNorm after every Conv** — stabilizes training, acts as implicit regularizer
- **No bias in Conv layers** — redundant when followed by BatchNorm

---

## Training Pipeline

### Regularization Stack

| Technique | Purpose |
|-----------|---------|
| BatchNormalization | Training stability, implicit regularization |
| Dropout (0.3 / 0.15) | Prevent co-adaptation of neurons |
| Label Smoothing (0.1) | Prevent overconfidence on noisy genre boundaries |
| SpecAugment | Data augmentation via frequency/time masking |
| Weight Decay (1e-4) | L2 regularization via AdamW |

### Optimization

```
Optimizer : AdamW (lr=0.001, weight_decay=1e-4)
Scheduler : CosineAnnealingLR (T_max=50)
Loss      : CrossEntropyLoss (label_smoothing=0.1)
Grad Clip : max_norm=1.0
```

### Training Configuration

```yaml
epochs      : 50 (early stopping, patience=7)
batch_size  : 32
val_split   : 15%
test_split  : 15%
seed        : 42
```

### Colab Safety Features

- Checkpoint saved every 10 epochs → Google Drive
- Best model saved on val_acc improvement
- Resume from latest checkpoint on session reconnect
- CUDA OOM handling with automatic batch skip

---

## Dataset

**Free Music Archive (FMA) — Small Subset**

| Property | Value |
|----------|-------|
| Total tracks | 8,000 |
| Genres | 8 (perfectly balanced) |
| Tracks per genre | 1,000 |
| Clip duration | 30 seconds |
| Audio quality | 128kbps MP3 |
| Null labels | 0 |

**Genre distribution:**

```
Hip-Hop        ████████████ 1000
Pop            ████████████ 1000
Folk           ████████████ 1000
Experimental   ████████████ 1000
Rock           ████████████ 1000
International  ████████████ 1000
Electronic     ████████████ 1000
Instrumental   ████████████ 1000
```

FMA was chosen over GTZAN for its scale, label quality, and real-world genre diversity. Perfect class balance eliminates the need for weighted loss functions.

---

## Results

**Training stopped at epoch 25 (early stopping)**

| Metric | Value |
|--------|-------|
| Test Accuracy | **60.63%** |
| Macro F1 Score | **0.59** |
| Weighted F1 Score | **0.59** |

### Per-Genre Performance

| Genre | Precision | Recall | F1 |
|-------|-----------|--------|----|
| Hip-Hop | 0.76 | 0.78 | **0.77** |
| Rock | 0.69 | 0.71 | **0.70** |
| Electronic | 0.56 | 0.78 | 0.65 |
| Folk | 0.58 | 0.74 | 0.65 |
| International | 0.60 | 0.65 | 0.63 |
| Instrumental | 0.69 | 0.47 | 0.56 |
| Experimental | 0.50 | 0.52 | 0.51 |
| Pop | 0.44 | 0.22 | **0.29** |

## Model Card

See full model details, limitations, and intended use on Hugging Face:
https://huggingface.co/jessitoi/genre-cnn-fma-small

### Key Observations

- **Hip-Hop and Rock** perform best — both have highly distinctive rhythmic and timbral signatures that manifest clearly in spectrograms
- **Pop** is the weakest class (F1=0.29) — by definition, Pop borrows elements from other genres, making it spectrally ambiguous
- **Instrumental** confuses with Experimental — both lack vocal signatures, the primary differentiator for many genre boundaries
- Validation loss shows high variance, indicating the model would benefit from a larger batch size and lower learning rate

### Confusion Matrix

![Confusion Matrix](confusion_matrix.png)

### Training Curves

![Training and Validation Loss](loss_graph.png)

---

## Project Structure

```
CNN-Based-Genre-Recognition-FMA/
│
├── .agents/
│   └── GEMINI.md              ← Agent context for Antigravity IDE
│
├── ai/
│   ├── config.yaml            ← All hyperparameters (single source of truth)
│   ├── requirements.txt       ← Colab dependencies
│   ├── notebooks/
│   │   └── train.ipynb        ← Google Colab training notebook
│   └── src/
│       ├── dataset.py         ← FMA loader, mel-spectrogram pipeline
│       ├── model.py           ← GenreCNN architecture
│       ├── train.py           ← Training loop, checkpointing, early stopping
│       └── evaluate.py        ← Test evaluation, confusion matrix
│
├── backend/
│   ├── main.py                ← FastAPI app (/predict, /genres, /health)
│   ├── model_loader.py        ← Model loading + mock mode
│   ├── utils.py               ← Audio preprocessing
│   ├── schemas.py             ← Pydantic response models
│   └── requirements.txt       ← Backend dependencies
│
├── web/                       ← Next.js demo frontend
│
├── data/                      ← Dataset (gitignored)
├── models/                    ← Trained weights (gitignored)
└── docs/                      ← Extended documentation
```

---

## Quick Start

### 1. Train on Google Colab

```
1. Upload fma_small.zip and fma_metadata.zip to Google Drive under:
   CNN-Based-Genre-Recognition-FMA/data/

2. Open ai/notebooks/train.ipynb in Google Colab
3. Set Runtime → T4 GPU
4. Run all cells
```

### 2. Run Backend

```bash
cd backend
pip install -r requirements.txt
uvicorn main:app --reload
# API available at http://localhost:8000
# Swagger docs at http://localhost:8000/docs
```

### 3. Run Frontend

```bash
cd web
npm install
npm run dev
# UI available at http://localhost:3000
```

---

## Tech Stack

| Layer | Technology |
|-------|-----------|
| Audio Processing | Librosa |
| Deep Learning | PyTorch |
| Data Augmentation | torchaudio SpecAugment |
| Training Infrastructure | Google Colab T4 |
| Backend API | FastAPI + Uvicorn |
| Frontend | Next.js + Tailwind CSS |

---

## Documentation

- [`docs/architecture.md`](docs/architecture.md) — System architecture and data flow
- [`docs/model.md`](docs/model.md) — CNN design rationale, layer-by-layer analysis
- [`docs/dataset.md`](docs/dataset.md) — FMA dataset analysis, preprocessing details
- [`docs/training.md`](docs/training.md) — Training pipeline, optimization strategy
- [`docs/results.md`](docs/results.md) — Full evaluation results and analysis
- [`docs/api.md`](docs/api.md) — API reference with curl examples

---

## Academic Context

This project was developed as a graduation project for the **Applied Neural Networks** course at **Uludağ University, Department of Computer Engineering**.

The primary research question: *Can CNN architectures designed for visual recognition tasks be effectively repurposed for audio classification via spectrogram representation?*

**Answer: Yes — with 60.63% test accuracy across 8 balanced genre classes using a custom CNN trained from scratch, without any pretrained weights or transfer learning.**

---

## License

MIT License — see [LICENSE](LICENSE) for details.