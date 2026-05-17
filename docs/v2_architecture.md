# V2 System Architecture

## Overview
The **V2 Architecture** extends the original design by integrating local development workflows directly with persistent cloud storage via Google Drive for Desktop. This allows developers to work locally on their source code while training and evaluating on the massive Free Music Archive (FMA) dataset hosted in the cloud, completely eliminating local disk bottlenecks.

The system is split into three layers:
1. **Cloud Data Layer (Google Drive)**: Hosts the raw audio files (`fma_small.zip`), metadata (`fma_metadata.zip`), preprocessed Mel-spectrogram files (`.npy` format), and trained models.
2. **Virtual Filesystem Bridge (Directory Junctions)**: Mounts the Google Drive storage locally as a standard filesystem path.
3. **AI & API Components (Local)**: Executes the PyTorch training and evaluation pipelines locally, while accessing/writing files dynamically via the bridge.

---

## Data Flow Diagram

```text
+------------------------------------+
|  Google Drive (Cloud Storage)      |
|  - data/raw/fma_metadata/          |
|  - data/processed/spectrograms/    |
|  - models/best_model_v2.pth        |
+------------------------------------+
                   |
                   | (Google Drive for Desktop - G:\ disk mount)
                   v
+------------------------------------+
|   Local Workspace Junctions        |
|   (Virtual Directory Symlinks)     |
|   - ./data  ==> G:\My Drive\...    |
|   - ./models ==> G:\My Drive\...   |
+------------------------------------+
         |                  ^
         v                  |
+-------------------+ +-------------------+
|  Local Execution  | |  Local Execution  |
|  (train.py)       | |  (evaluate.py)    |
+-------------------+ +-------------------+
         |                  |
         +---------+--------+
                   |
                   v
+------------------------------------+
|         FastAPI Backend            |
|       (backend/main.py)            |
|  - ModelManager (Real V2 Model)    |
|  - Exposes: /predict, /genres      |
+------------------------------------+
```

---

## Hybrid Cloud-Local Workspace

By utilizing Windows Directory Junctions (symlinks), the local Python scripts run exactly as if the 8GB dataset was stored on the local SSD, while the operating system and the Google Drive client handle streaming files on-demand in the background.

### Local Setup Script
The virtual-to-local bridging is executed in PowerShell as follows:

```powershell
# Rename local placeholders
Rename-Item -Path ./data -NewName data_local_backup
Rename-Item -Path ./models -NewName models_local_backup

# Create junctions pointing to the active Google Drive mount
New-Item -ItemType Junction -Path ./data -Value "G:\My Drive\Okul\CNN-Based-Genre-Recognition-FMA\data"
New-Item -ItemType Junction -Path ./models -Value "G:\My Drive\Okul\CNN-Based-Genre-Recognition-FMA\models"
```

---

## Component Interactions (V2)

- **Config Handler ([config.yaml](file:///d:/Software/cnn-based-genre-recognition-fma/ai/config.yaml))**: Holds the single source of truth parameters. Serves as config loader for local train/evaluation scripts as well as Google Colab environments.
- **Model Loader ([model_loader.py](file:///d:/Software/cnn-based-genre-recognition-fma/backend/model_loader.py))**: Instantiates the `GenreCNN` architecture and loads `models/best_model_v2.pth` (linked from Google Drive). If the weights are missing, it falls back to a simulated *mock* mode so developers can run and test the frontend UI or API endpoints instantly without accessing cloud files.
- **Audio Preprocessing ([utils.py](file:///d:/Software/cnn-based-genre-recognition-fma/backend/utils.py))**: Extracts Mel-spectrograms on the fly from uploaded client MP3 audio files, performs log-scaling (dB) and sample-wise normalization, creating a tensor of shape `(1, 1, 128, T)` compatible with the `GenreCNN` V2 classifier.
