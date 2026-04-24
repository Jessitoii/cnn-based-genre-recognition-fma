# AI Source Code Modules

This module encompasses the core deep learning operations of our music genre recognition API utilizing PyTorch and Librosa. 

## Module-by-Module Explanation

### `dataset.py`
Oversees data processing and the PyTorch Dataset interface structure.
- Implements audio preprocessing chains translating `.mp3`/`.wav` frames to normalized Mel-spectrogram matrices.
- Returns customized `torch.utils.data.Dataset` mapping features directly to their corresponding categorical integer variables.

### `evaluate.py`
Deploys standardized mechanisms for measuring trained model inference capacity over unobserved data points. 
- Iterates over validation/test splits computing overall Accuracy metrics, precision, recall, and plotting confusion matrices for further scrutiny.

### `explore.py`
Serves as an ad-hoc debugging and explorative analytics notebook script mapping specific spectrogram renderings for direct visual examination and basic integrity checks.

### `model.py`
Exposes the PyTorch model definitions.
- Specifies the `GenreCNN` class extending `nn.Module`.
- Outlines the layers: Conv2D, MaxPooling, BatchNorm, Dropout, and final linear sequential mapping configurations aligned dynamically to the configuration parameters.

### `train.py`
Executes main iterative training routines.
- Injects `dataset.py` elements into native PyTorch dataloaders.
- Dictates epoch loops updating network weights backward utilizing an Adam Optimizer over generalized categorical cross-entropy loss criteria.
- Persists checkpoints frequently and measures relative validation offsets natively.
