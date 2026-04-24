# Training Pipeline

## Overview
The training pipeline for the CNN is located in `ai/src/train.py`. It leverages the AdamW optimizer along with Cosine Annealing learning rate scheduling to ensure fast convergence while preventing catastrophic forgetting.

## Early Stopping
To prevent the model from overfitting on the limited `fma_small` dataset, early stopping is implemented. The validation loss is monitored at the end of each epoch. If the validation loss fails to improve for 7 consecutive epochs (`patience = 7`), training is halted prematurely. 

## Checkpointing Strategy
Given that the primary training environment is Google Colab (which is prone to disconnections and timeouts), robust checkpointing is necessary:
1. **Best Model Checkpoint**: Saved as `best_model.pth` whenever a new low in validation loss is reached. Contains the model state, optimizer state, epoch number, and config.
2. **Periodic Checkpoint**: Saved every 10 epochs as `epoch_XXX.pth` to ensure there is a fallback point even if the system crashes midway. Checkpoints are automatically synced to Google Drive when `colab: true` is provided.

## CosineAnnealingLR
We employ `CosineAnnealingLR` (from `torch.optim.lr_scheduler`) which gradually decays the learning rate following a cosine curve from the initial `0.001` to `0` over the max number of epochs. 
**Why?**
1. **Initial Fast Learning**: Helps the optimizer settle into an optimal basin quickly.
2. **Fine-Tuning**: Towards the end of the epochs, the slow decay helps fine-tune weights precisely without over-shooting the local minimum.
