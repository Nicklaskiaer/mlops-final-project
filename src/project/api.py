import torch
import uvicorn
import shutil
from fastapi import FastAPI, UploadFile, File, HTTPException
from pathlib import Path
from src.project.model import HubertClassifier

# Initialize FastAPI
app = FastAPI(title="Infant Cry Audio Classifier")

# Global variables for model storage
model_instance = None
MODEL_PATH = Path("models/checkpoints/best_hubert_model.pt")

# Hardcoded classes based on your sorted folder names (alphabetical order)
# This ensures we map index 0 -> "belly_pain", etc. correctly without loading the dataset.
CLASS_NAMES = ["belly pain", "burping", "cold_hot", "discomfort", "hungry", "lonely", "scared", "tired"]


@app.on_event("startup")
def load_model():
    """
    Load the model into memory when the server starts.
    """
    global model_instance

    # 1. Device selection (M1/CUDA/CPU)
    if torch.cuda.is_available():
        device = "cuda"
    elif torch.backends.mps.is_available():
        device = "mps"
    else:
        device = "cpu"

    print(f"Loading model on {device}...")

    try:
        # Initialize the architecture (Must match training!)
        model = HubertClassifier(
            model_name="ntu-spml/distilhubert",  # TODO: use .env var instead
            num_labels=len(CLASS_NAMES),
        )

        # Load weights
        if MODEL_PATH.exists():
            state_dict = torch.load(MODEL_PATH, map_location=device)
            model.load_state_dict(state_dict)
            model.to(device)
            model.eval()
            model_instance = model
            print("Model loaded successfully!")
        else:
            print(f"WARNING: Checkpoint not found at {MODEL_PATH}. API will fail on inference.")

    except Exception as e:
        print(f"Error loading model: {e}")


@app.post("/predict")
async def predict(file: UploadFile = File(...)):
    """
    Endpoint to predict the class of an uploaded audio file.
    """
    if model_instance is None:
        raise HTTPException(status_code=503, detail="Model not loaded")

    # 1. Save uploaded file temporarily
    temp_file = Path(f"temp_{file.filename}")
    try:
        with temp_file.open("wb") as buffer:
            shutil.copyfileobj(file.file, buffer)

        # 2. Run Inference using our Model's robust predict method
        # This handles the resampling/preprocessing internally
        result = model_instance.predict(temp_file)

        # 3. Add human-readable label
        predicted_idx = result["predicted_label_idx"]
        label_name = CLASS_NAMES[predicted_idx] if predicted_idx < len(CLASS_NAMES) else "Unknown"

        return {
            "filename": file.filename,
            "prediction": label_name,
            "confidence": f"{result['confidence']:.2%}",
            "details": result,
        }

    except Exception as e:
        raise HTTPException(status_code=500, detail=str(e))

    finally:
        # Cleanup temp file
        if temp_file.exists():
            temp_file.unlink()


@app.get("/health")
def health_check():
    return {"status": "ok", "model_loaded": model_instance is not None}


if __name__ == "__main__":
    # Allow running this script directly for debugging
    uvicorn.run(app, host="0.0.0.0", port=8000)
