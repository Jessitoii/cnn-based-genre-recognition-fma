# Dataset Documentation

## FMA Dataset Analysis
The project uses the "Small" subset of the Free Music Archive (FMA) dataset.
- Total Tracks: 8,000 (appx 8 GiB)
- Subsets Chosen: `fma_small` to easily fit into Google Colab T4 memory constraints while still having enough data for a CNN to learn descriptive features.

## Genre Distribution
The dataset is perfectly balanced across 8 core genres, providing 1,000 tracks per genre:
1. Electronic
2. Experimental
3. Folk
4. Hip-Hop
5. Instrumental
6. International
7. Pop
8. Rock

## Mel-Spectrogram Parameters & Theory
Audio classification often yields better results by treating sound as an image via a spectrogram. We use the log-mel spectrogram.

**Chosen Parameters (from `config.yaml`)**:
- `sample_rate`: 22050 Hz (industry standard baseline for audio reduction)
- `duration`: 30 seconds per clip (standardized for the entire fma_small dataset)
- `n_fft`: 2048 (length of the FFT window)
- `hop_length`: 512 (number of samples between successive frames)
- `n_mels`: 128 (number of Mel bands to generate)

**Theory**:
1. **Fourier Transform**: Calculates the frequency magnitudes across small time windows (controlled by `n_fft` and `hop_length`).
2. **Mel Scale**: Frequencies are warped into the Mel scale, resembling human auditory perception (closer resolution at lower pitches).
3. **Log Scale**: We convert power to decibels (dB), matching human perception of loudness.
4. **Resulting Target Shape**: `(1, 128, 1292)` tensors representing image-like temporal and frequency-domain correlations for the CNN.
