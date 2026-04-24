# Model Documentation

## CNN Architecture Decision Rationale
We selected a custom Convolutional Neural Network (`GenreCNN`) over recurrent architectures. CNNs excel at recognizing spatial patterns (like harmonic structures, percussive transients) which become visually evident on Mel-spectrograms. The model is lightweight to enable real-time CPU inference on the backend and stable training on standard GPU hardware (Colab T4).

## Layer-by-Layer Explanation
- **Feature Extractor (4 Conv Blocks)**:
  - Block 1: `Conv2D(1 -> 32)` capturing basic low-level features like edges or frequency lines.
  - Block 2: `Conv2D(32 -> 64)` capturing mid-level textures.
  - Block 3: `Conv2D(64 -> 128)` detecting structure.
  - Block 4: `Conv2D(128 -> 256)` detecting complex semantic classes.
  - *Each block employs a `3x3` kernel, `BatchNorm2D`, `ReLU`, and `MaxPool2D` to downsample spatial dimensions.*

- **Global Average Pooling**:
  - `AdaptiveAvgPool2d(1)` flattens the time and frequency space into a single feature vector representing the overall audio context without relying on fixed temporal bounds.

- **Dense Classifier**:
  - `Linear(256 -> 128)` processing global context features.
  - `Dropout`
  - `Linear(128 -> 8)` outputting final logits for 8 genres.

## Hyperparameter Choices
- **Classes**: 8 (matches FMA setup).
- **Dropout Rate**: `0.3` (prevent over-reliance on limited training samples).
- **Optimizer Learning Rate**: `0.001` with standard AdamW weight decay `0.0001`.

## Regularization Techniques
Given limited data (8k tracks), aggressive regularization is used:
1. **BatchNorm**: Applied immediately after convolutions to stabilize distributions and implicitly acts as a regularizer.
2. **Dropout**: Dropping 30% of weights to prevent overfitting in the Dense classifier.
3. **Label Smoothing**: Softens the rigorous one-hot labels to `0.9` for the true class and `0.1/7` for the remaining. Handled in `CrossEntropyLoss(label_smoothing=0.1)`. Reduces model over-confidence.
4. **SpecAugment**: Found in `train.py`. On-the-fly data augmentation masking bands of frequency and chunks of time (up to 27 freq bins and 100 time frames). Forces model to learn robust representations.
