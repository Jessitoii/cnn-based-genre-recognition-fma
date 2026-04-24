# System Architecture

## Overview
The CNN-Based Genre Recognition project is structured into two main components:
1. **AI Pipeline (`ai/`)**: Handles data processing (FMA dataset), model training, evaluation, and checkpointing. 
2. **Backend API (`backend/`)**: A FastAPI-based application that serves the trained CNN inference and exposes an HTTP interface for predicting genres of audio files.

## Data Flow Diagram
```text
+---------------------+     +--------------------------+
|  FMA Dataset (MP3)  | --> | Data Preprocessing (NPY) | (ai/src/dataset.py)
+---------------------+     +--------------------------+
                                     |
                                     v
+---------------------+     +--------------------------+
|  Training Pipeline  | <-- | Mel-spectrogram Tensors  |
|  (ai/src/train.py)  |     | (1, 128, T) shape        |
+---------------------+     +--------------------------+
          |
          v
+---------------------+
| Saved Checkpoints   | (models/checkpoints)
| (models/best.pth)   |
+---------------------+
          |
          v
+---------------------+     +--------------------------+
|  FastAPI Backend    | <-- | Client Audio Upload      |
| (backend/main.py)   | --> | (JSON Genre Prediction)  |
+---------------------+     +--------------------------+
```

## Component Interactions
- **AI Component (`ai/`)**: Defines the PyTorch data loaders (`FMADataset`), the CNN architecture (`GenreCNN`), and the training loop (`train()`). Uses `config.yaml` to orchestrate settings.
- **Backend Component (`backend/`)**: Uses `ModelManager` to load the saved `.pth` file or run in mock mode if not present. Receives audio file, uses Librosa to compute the Mel-spectrogram on the fly, and uses the AI model.
