    **CNN-Based-Genre-Recognition-FMA**

This repository contains a deep learning-based system designed for the automated organization of large-scale digital music libraries. By leveraging the **Free Music Archive (FMA)** dataset, the project provides a robust solution for identifying musical genres through computer vision techniques applied to audio signals.

**Technical Overview:**

*   **Feature Extraction**: Utilizing the **Librosa** library to transform 1D audio signals into 2D **Mel-Spectrograms**, capturing the frequency-time intensity as "heat maps" that mimic human auditory perception.
    
*   **Architecture**: A **Convolutional Neural Network (CNN)** architecture is employed to analyze the visual patterns within the spectrograms, such as rhythmic structures, tonal characteristics, and timbre.
    
*   **Dataset**: Focuses on the **FMA dataset** due to its high-quality audio samples and balanced genre diversity, offering a more realistic challenge compared to legacy datasets.
    
*   **Tech Stack**: Built with **Python**, **PyTorch/Keras**, **NumPy**, and **Pandas** for efficient matrix operations and model training.
    

**Project Pipeline:**

1.  **Audio Preprocessing**: Normalization and fixed-length windowing of audio files.
    
2.  **Spectrogram Generation**: Conversion of signal data into numerical 2D matrices.
    
3.  **Model Training**: CNN pattern recognition for genre classification.
    
4.  **Evaluation**: Performance assessment using Accuracy and Loss metrics on unseen data.