# Music Genre Recognition Dataset

This directory contains the dataset used for training and evaluating the CNN-based music genre classification model. We use the **Free Music Archive (FMA)** dataset, specifically the balanced `fma_small` subset.

## Dataset Structure

The project expects the following structure within the `data/` directory:

```text
data/
├── raw/
│   ├── fma_metadata/
│   │   ├── tracks.csv          # Core metadata file
│   │   ├── genres.csv          # Genre hierarchy
│   │   └── ...
│   └── fma_small/
│       ├── 000/                # 1000 sub-folders (000-999)
│       │   ├── 000002.mp3      # 30-second audio clips
│       │   └── ...
│       ├── 001/
│       └── ...
└── processed/
    └── spectrograms/
        ├── 000002.npy          # Precomputed Mel-spectrograms
        ├── 000005.npy
        └── ...
```

### Components

1.  **`fma_small`**: A subset of the FMA dataset consisting of 8,000 tracks, each 30 seconds long, balanced across 8 genres: Electronic, Experimental, Folk, Hip-Hop, Instrumental, International, Pop, and Rock.
2.  **`fma_metadata`**: Contains track information, genre annotations, and features. The `tracks.csv` file is essential for the `FMADataset` loader.
3.  **`processed/spectrograms`**: This directory stores the output of the preprocessing step. Audio files are converted into Mel-spectrograms (128 mel bands) and saved as `.npy` files to accelerate training by avoiding redundant STFT computations.

## Preprocessing Details

-   **Output Format**: `.npy` (NumPy binary).
-   **Naming Convention**: `track_id.npy` (e.g., `000002.npy`).
-   **Content**: Log-scaled Mel-spectrogram of shape `(128, T)` where $T \approx 1292$ for 30s clips.

## How to Set Up the Dataset

1.  **Download**:
    -   Download `fma_metadata.zip` and `fma_small.zip` from the [official FMA repository](https://github.com/mdeff/fma).
2.  **Extract**:
    -   Create a `data/raw/` folder.
    -   Extract `fma_metadata.zip` into `data/raw/fma_metadata/`.
    -   Extract `fma_small.zip` into `data/raw/fma_small/`.
3.  **Preprocess**:
    -   Run the preprocessing script to generate spectrograms. This is typically handled automatically in the training notebook or via `ai/src/dataset.py`.
    -   Spectrograms will be saved to `data/processed/spectrograms/`.

---
> [!NOTE]
> The `data/` directory is excluded from version control (Git) to keep the repository lightweight. Ensure local backups are maintained.
