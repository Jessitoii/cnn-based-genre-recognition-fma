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

# Load config and model
cfg = load_config()
model_manager = ModelManager(cfg)

# CORS setup for Next.js frontend
app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],  # Adjust this in production
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)


@app.get("/health", response_model=HealthResponse)
async def health_check():
    return {"status": "ok", "model_mode": model_manager.mode}


@app.get("/genres", response_model=GenreListResponse)
async def get_genres():
    return {"genres": model_manager.genres}


@app.post("/predict", response_model=PredictionResponse)
async def predict(file: UploadFile = File(...)):
    if not file.filename.endswith((".mp3", ".wav", ".ogg")):
        raise HTTPException(status_code=400, detail="Unsupported audio format")

    # Create a temporary file to save the upload
    with tempfile.NamedTemporaryFile(
        delete=False, suffix=os.path.splitext(file.filename)[1]
    ) as tmp:
        try:
            shutil.copyfileobj(file.file, tmp)
            tmp_path = tmp.name
        finally:
            file.file.close()

    try:
        # Preprocess
        input_tensor = process_audio(tmp_path, cfg)

        # Inference
        genre, confidence, all_scores = model_manager.predict(input_tensor)

        return {"genre": genre, "confidence": confidence, "all_scores": all_scores}
    except Exception as e:
        raise HTTPException(status_code=500, detail=f"Inference failed: {str(e)}")
    finally:
        # Clean up temp file
        if os.path.exists(tmp_path):
            os.remove(tmp_path)


if __name__ == "__main__":
    import uvicorn

    uvicorn.run(app, host="0.0.0.0", port=8000)
