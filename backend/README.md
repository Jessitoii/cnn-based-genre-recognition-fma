# Backend Service API

The backend provides a stateless RESTful JSON interface engineered over FastAPI bridging the web inference portal natively securely with the PyTorch predictive operations payload. 

## API Endpoints

### 1. `GET /health`
Returns the status of the API and its underlying model loading condition (whether actual or mock fallback context).

**Response Example:**
```json
{
  "status": "ok",
  "model_mode": "mock"
}
```

### 2. `GET /genres`
Fetches a chronological vector payload holding the absolute array indexing of music genres configured into the predictive architecture.

**Response Example:**
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

### 3. `POST /predict`
The main ingestion point. Takes `multipart/form-data` uploads comprising `.mp3`, `.ogg`, or `.wav` streams, evaluates the array via internal model tensors sequentially, and propagates relative inference percentiles formatted gracefully. 

**Request:** `file` (binary form-data)
**Response Example:**
```json
{
  "genre": "Rock",
  "confidence": 0.89,
  "all_scores": {
    "Rock": 0.89,
    "Pop": 0.05,
    "Folk": 0.03,
    "Electronic": 0.02,
    "Experimental": 0.01,
    "Hip-Hop": 0.00,
    "Instrumental": 0.00,
    "International": 0.00
  }
}
```

## Mock Mode Explanation
This module natively provides automatic failure compensation via "Mock Mode". In cases where the best `models/best_model.pth` target is absent or inference engines (like Torch runtime constraints) fault unexpectedly—rather than terminating connection channels, the fallback model assigns randomized probabilistic outcomes to genres explicitly signaling `model_mode: "mock"` back to developers ensuring uninterrupted frontend interface iteration unblocked concurrently.
