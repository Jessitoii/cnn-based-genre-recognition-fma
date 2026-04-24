# CNN-Based Genre Recognition — Project Context

## Project Overview
This repository contains an academic project for an Applied Neural Networks course. The primary goal is to classify music genres using Convolutional Neural Networks (CNNs) trained on Mel-Spectrogram representations of audio tracks from the Free Music Archive (FMA) dataset.

## Architecture
```text
+-------------------+      +--------------------+      +-------------------+
|                   |      |                    |      |                   |
|   Web Frontend    | <--> |   FastAPI Backend  | <--> |   CNN PyTorch     |
|   (Next.js)       |      |   (REST API)       |      |   Model           |
|                   |      |                    |      |                   |
+-------------------+      +--------------------+      +-------------------+
        |                            |                           |
        v                            v                           |
  User Interface             Backend processing              Trained on Colab T4 Cloud GPU
  for Uploading              Feature Extraction              Using fma_small dataset
  Audio Files                (Mel-Spectrogram)               (Saved in Google Drive)
```

## Tech Stack
- **AI / ML**: Python, PyTorch, Torchaudio, Librosa, NumPy, Matplotlib
- **Backend API**: Python, FastAPI, Uvicorn, Pydantic
- **Frontend Web**: Next.js, React, TypeScript, Tailwind CSS

## Setup Instructions

### Prerequisites
- Python 3.10+
- Node.js 18+

### 1. Model Training
Please refer to [ai/README.md](ai/README.md) for details on training the model using Google Colab.

### 2. Backend Setup
Navigate to the backend directory, install the dependencies, and start the development server:

```bash
cd backend
pip install -r requirements.txt
python main.py
```
*Note: The backend runs on `localhost:8000`. If you do not have a trained model, the backend relies on a mock mode dynamically responding with fake predictions.*

### 3. Frontend Setup
Navigate to the web directory, install the packages, and run the Next.js frontend:

```bash
cd web
npm install
npm run dev
```
*Note: The frontend runs on `localhost:3000`.*

## Results

**Placeholder:** 
| Metric | Value |
|--------|-------|
| Accuracy | TBA |
| Loss | TBA |
| F1 Score | TBA |