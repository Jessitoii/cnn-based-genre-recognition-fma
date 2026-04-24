"""
Model management module for the backend.
Handles loading the PyTorch GenreCNN model with a fallback to a mock
inference mode if the model weights are not found or fail to load.
"""

import torch
import os
import sys
from pathlib import Path
import logging

# Add ai/src to path to import GenreCNN for standard model loading
sys.path.append(str(Path(__file__).parent.parent / "ai" / "src"))
try:
    from model import GenreCNN
except ImportError:
    # Fallback if directory structure differs in certain environments
    from ai.src.model import GenreCNN

logger = logging.getLogger(__name__)


class ModelManager:
    """
    Manages model initialization, device placement, and inference logic.
    Supports both real model inference and a simulated mock mode for testing/local development.
    """

    def __init__(self, cfg):
        """
        Initialize the ModelManager by attempting to load the trained weights.

        Args:
            cfg (dict): Configuration dictionary containing model hyperparameters and file paths.
        """
        self.cfg = cfg
        self.device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
        self.model_path = Path(__file__).parent.parent / cfg["paths"]["best_model"]
        self.model = None
        self.mode = "mock"  # Default to mock
        self.genres = cfg["data"]["genres"]

        # Attempt to load the production model from disk
        if self.model_path.exists():
            try:
                self.model = GenreCNN(
                    num_classes=cfg["model"]["num_classes"],
                    dropout=cfg["model"]["dropout"],
                )

                # Load state dict based on the mapping compatible with the current device
                state_dict = torch.load(self.model_path, map_location=self.device)

                # Check if it's a full checkpoint or just weights
                if "model_state" in state_dict:
                    self.model.load_state_dict(state_dict["model_state"])
                else:
                    self.model.load_state_dict(state_dict)

                self.model.to(self.device)
                self.model.eval()
                self.mode = "real"
                logger.info(f"Loaded real model from {self.model_path}")
            except Exception as e:
                logger.error(f"Failed to load model: {e}. Switching to mock mode.")
                self.mode = "mock"
        else:
            logger.warning(
                f"Model weights file not found at {self.model_path}. Using mock mode."
            )
            self.mode = "mock"

    def predict(self, input_tensor):
        """
        Perform genre prediction on an input Mel-spectrogram tensor.

        Args:
            input_tensor (torch.Tensor): Preprocessed audio tensor of shape (1, 1, 128, T).

        Returns:
            tuple: (predicted_genre_name, confidence_score, all_genre_probabilities)
                   where predicted_genre_name is a string, confidence_score is a float,
                   and all_genre_probabilities is a dictionary mapping genre names to scores.
        """
        if self.mode == "real":
            # Real model inference path
            with torch.no_grad():
                input_tensor = input_tensor.to(self.device)
                logits = self.model(input_tensor)
                probs = torch.softmax(logits, dim=1).squeeze()

                confidence, class_idx = torch.max(probs, dim=0)
                genre = self.genres[class_idx.item()]

                all_scores = {
                    self.genres[i]: float(probs[i]) for i in range(len(self.genres))
                }

                return genre, float(confidence), all_scores
        else:
            # Mock mode: returns slightly randomized but deterministic-ish scores
            # Use for testing frontend without requiring a GPU or FMA dataset
            import random

            scores = [random.random() for _ in self.genres]
            sum_scores = sum(scores)
            probs = [s / sum_scores for s in scores]

            max_val = max(probs)
            max_idx = probs.index(max_val)

            all_scores = {self.genres[i]: probs[i] for i in range(len(self.genres))}

            return self.genres[max_idx], max_val, all_scores
