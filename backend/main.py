"""
Main FastAPI application for Music Genre Classification.
Provides REST endpoints for health checks, getting available genres,
and predicting the genre of uploaded audio files.
"""

import os
import shutil
import tempfile
from fastapi import FastAPI, UploadFile, File, HTTPException
from fastapi.middleware.cors import CORSMiddleware

try:
    from .schemas import PredictionResponse, GenreListResponse, HealthResponse
    from .utils import load_config, process_audio
    from .model_loader import ModelManager
except ImportError:
    from schemas import PredictionResponse, GenreListResponse, HealthResponse
    from utils import load_config, process_audio
    from model_loader import ModelManager

app = FastAPI(title="Genre Classification API")

# Initialize global configuration and model manager
cfg = load_config()
model_manager = ModelManager(cfg)

# CORS setup for permitting cross-origin requests (e.g. from a Next.js frontend)
app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],  # Note: Adjust this to specific domains in production
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)


@app.get("/health", response_model=HealthResponse)
async def health_check():
    """
    Check the status of the API and the model loading mode.

    Returns:
        dict: Status message and current model mode (loaded or mock).
    """
    return {"status": "ok", "model_mode": model_manager.mode}


@app.get("/genres", response_model=GenreListResponse)
async def get_genres():
    """
    Retrieve the list of supported music genres.

    Returns:
        dict: A list of genre names known to the model.
    """
    return {"genres": model_manager.genres}


@app.post("/predict", response_model=PredictionResponse)
async def predict(file: UploadFile = File(...)):
    """
    Predict the genre of an uploaded audio file.

    This endpoint saves the file temporarily, converts it to a Mel-spectrogram,
    runs model inference, and returns the predicted genre and confidence.

    Args:
        file (UploadFile): The audio file to classify (MP3, WAV, or OGG).

    Returns:
        dict: Predicted genre, confidence score, and all genre scores.

    Raises:
        HTTPException: If the file format is unsupported or inference fails.
    """
    if not file.filename.endswith((".mp3", ".wav", ".ogg")):
        raise HTTPException(status_code=400, detail="Unsupported audio format")

    # Create a temporary file to safely store the uploaded binary data for librosa processing
    with tempfile.NamedTemporaryFile(
        delete=False, suffix=os.path.splitext(file.filename)[1]
    ) as tmp:
        try:
            shutil.copyfileobj(file.file, tmp)
            tmp_path = tmp.name
        finally:
            file.file.close()

    try:
        # Preprocess the audio into the required tensor format
        input_tensor = process_audio(tmp_path, cfg)

        # Execute model inference
        genre, confidence, all_scores = model_manager.predict(input_tensor)

        return {"genre": genre, "confidence": confidence, "all_scores": all_scores}
    except Exception as e:
        raise HTTPException(status_code=500, detail=f"Inference failed: {str(e)}")
    finally:
        # Ensure the temporary file is deleted even if inference fails
        if os.path.exists(tmp_path):
            os.remove(tmp_path)


if __name__ == "__main__":
    import uvicorn

    # Run the application using Uvicorn server
    uvicorn.run(app, host="0.0.0.0", port=8000)
