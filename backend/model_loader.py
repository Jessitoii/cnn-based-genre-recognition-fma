import torch
import os
import sys
from pathlib import Path
import logging

# Add ai/src to path to import GenreCNN
sys.path.append(str(Path(__file__).parent.parent / "ai" / "src"))
try:
    from model import GenreCNN
except ImportError:
    # Fallback if structure is different
    from ai.src.model import GenreCNN

logger = logging.getLogger(__name__)


class ModelManager:
    def __init__(self, cfg):
        self.cfg = cfg
        self.device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
        self.model_path = Path(__file__).parent.parent / cfg["paths"]["best_model"]
        self.model = None
        self.mode = "mock"
        self.genres = cfg["data"]["genres"]

        if self.model_path.exists():
            try:
                self.model = GenreCNN(
                    num_classes=cfg["model"]["num_classes"],
                    dropout=cfg["model"]["dropout"],
                )
                self.model.load_state_dict(
                    torch.load(self.model_path, map_location=self.device)
                )
                self.model.to(self.device)
                self.model.eval()
                self.mode = "real"
                logger.info(f"Loaded model from {self.model_path}")
            except Exception as e:
                logger.error(f"Failed to load model: {e}. Switching to mock mode.")
                self.mode = "mock"
        else:
            logger.warning(
                f"Model file not found at {self.model_path}. Using mock mode."
            )
            self.mode = "mock"

    def predict(self, input_tensor):
        if self.mode == "real":
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
            import random

            scores = [random.random() for _ in self.genres]
            sum_scores = sum(scores)
            probs = [s / sum_scores for s in scores]

            max_val = max(probs)
            max_idx = probs.index(max_val)

            all_scores = {self.genres[i]: probs[i] for i in range(len(self.genres))}

            return self.genres[max_idx], max_val, all_scores
