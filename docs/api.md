# API Reference

The backend exposes a RESTful FastAPI service for interacting with the genre classifier.

## Global Setup
The backend runs on `http://localhost:8000`. Ensure that you have installed the requirements and executed `uvicorn main:app --reload`.

---

## `GET /health`
Validates if the API is running and indicates how the model is loaded (actual `.pth` file or mock mode fallback).

**Example Request:**
```bash
curl -X GET "http://localhost:8000/health"
```

**Example Response:**
```json
{
  "status": "ok",
  "model_mode": "loaded"
}
```

---

## `GET /genres`
Retrieves the list of genres supported natively by the underlying model.

**Example Request:**
```bash
curl -X GET "http://localhost:8000/genres"
```

**Example Response:**
```json
{
  "genres": [
    "Electronic",
    "Experimental",
    "Folk",
    "Hip-Hop",
    "Instrumental",
    "International",
    "Pop",
    "Rock"
  ]
}
```

---

## `POST /predict`
Accepts an audio file (`.mp3`, `.wav`, or `.ogg`), computes its Mel-spectrogram on the fly, and uses the GenreCNN to predict its class.

**Parameters:**
- `file` (multipart/form-data): The binary audio file.

**Example Request:**
```bash
curl -X POST "http://localhost:8000/predict" \
  -H "accept: application/json" \
  -H "Content-Type: multipart/form-data" \
  -F "file=@sample_song.mp3"
```

**Example Response:**
```json
{
  "genre": "Electronic",
  "confidence": 0.85,
  "all_scores": {
    "Electronic": 0.85,
    "Pop": 0.05
  }
}
```
