# CNN-Based Genre Recognition — Project Context

## Goal
Classify music genres using CNN on Mel-Spectrograms from the FMA dataset.
Academic project for Applied Neural Networks course.

## Dataset
- FMA Small (8k tracks, 8 genres, balanced)
- Local: data/ | Google Drive: CNN-Based-Genre-Recognition-FMA/data/

## Stack
- Python, PyTorch, Librosa, NumPy, Matplotlib
- Training: Google Colab (T4 GPU)
- Drive mount path: /content/drive/MyDrive/CNN-Based-Genre-Recognition-FMA/

## Project Structure
ai/src/        → dataset.py, model.py, train.py, evaluate.py
ai/notebooks/  → train.ipynb (Colab entry point)
ai/config.yaml → all hyperparameters
models/        → saved checkpoints (.gitignored)
data/          → raw + processed data (.gitignored)

## Key Decisions
- fma_small chosen for Colab T4 memory constraints
- Mel-spectrogram: 128 mel bands, 30s clips, hop_length=512
- CNN architecture: Conv2D → BatchNorm → Pool → Dropout → Dense
- Checkpointing to Drive every epoch (Colab disconnect risk)