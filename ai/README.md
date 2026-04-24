# AI Engine: Training Pipeline & Theory

This directory handles the training, evaluation, and iteration of the Convolutional Neural Network (CNN) used for music genre classification. 

## ML Pipeline Explanation
The core pipeline involves transforming 30-second audio clips into images (Mel-spectrograms), processing them in batches, mapping the data into tensors, and passing them through a CNN to classify between 8 distinct genres (Electronic, Experimental, Folk, Hip-Hop, Instrumental, International, Pop, Rock). 

## Mel-Spectrogram Theory
A Mel-spectrogram provides a visual representation of the audio signal over time but places frequencies on a mel scale, which accurately approximates human hearing perception. 
- **Sample Rate**: 22050 Hz
- **n_fft**: 2048
- **hop_length**: 512
- **n_mels**: 128

This configuration extracts dense acoustic features that are ideal for deep representation learning.

## CNN Architecture Details
Our architecture leverages visual feature extraction mapping to standard classification heads:
1. **Conv2D Blocks**: Sequential blocks composed of 2D Convolution layers applied across the spectrogram.
2. **BatchNorm & Pooling**: For feature scaling and dimensionality reduction between layers.
3. **Dropout**: Random neuron dropout regularization (Probability: 0.3) is instituted to avoid overfitting.
4. **Dense Layer**: Mapping flattened 1D features through a Multilayer Perceptron (MLP) into the logit probabilities for our 8 distinct genres.

## Training Configuration
Parameters are centralized under `config.yaml`. The choice of `fma_small` (8k tracks) ensures compatibility with local iterations and Google Colab T4 GPU constraints.
- **Batch Size**: 32
- **Epochs**: 50
- **Optimizer**: Adam (Learning Rate: 0.001)

## How to Run on Colab
1. Upload the `CNN-Based-Genre-Recognition-FMA/ai` and the dataset to your Google Drive (`/content/drive/MyDrive/CNN-Based-Genre-Recognition-FMA/`).
2. Open `ai/notebooks/train.ipynb` using Google Colab.
3. Set your runtime to use the **T4 GPU**.
4. Run all cells in the Jupyter Notebook to mount your drive, configure dependencies, start preprocessing, and initialize testing and evaluation. Checkpoints are iteratively written straight back to Google Drive to dodge potential session disconnects.
