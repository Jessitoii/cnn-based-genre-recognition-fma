"""
Pydantic schemas for API request and response validation.
Defines the structure of data exchanged between the client and server.
"""

from pydantic import BaseModel
from typing import List, Dict


class PredictionResponse(BaseModel):
    """
    Response schema for the /predict endpoint.

    Attributes:
        genre (str): The name of the predicted music genre.
        confidence (float): The probability score for the predicted genre.
        all_scores (Dict[str, float]): A dictionary containing scores for all supported genres.
    """

    genre: str
    confidence: float
    all_scores: Dict[str, float]


class GenreListResponse(BaseModel):
    """
    Response schema for the /genres endpoint.

    Attributes:
        genres (List[str]): A list of all genre names supported by the model.
    """

    genres: List[str]


class HealthResponse(BaseModel):
    """
    Response schema for the /health endpoint.

    Attributes:
        status (str): The operational status of the API (e.g., "ok").
        model_mode (str): Indicates if the model is currently running in "real" or "mock" mode.
    """

    status: str
    model_mode: str  # "real" or "mock"
