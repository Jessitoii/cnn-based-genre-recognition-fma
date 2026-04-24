from pydantic import BaseModel
from typing import List, Dict


class PredictionResponse(BaseModel):
    genre: str
    confidence: float
    all_scores: Dict[str, float]


class GenreListResponse(BaseModel):
    genres: List[str]


class HealthResponse(BaseModel):
    status: str
    model_mode: str  # "real" or "mock"
