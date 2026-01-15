import torch
import uvicorn
import shutil
from contextlib import asynccontextmanager
from fastapi import FastAPI, UploadFile, File, HTTPException
from pathlib import Path

# Assuming this import works in your environment
from src.project.model import HubertClassifier

# Global variables for model storage
model_instance = None
MODEL_PATH = Path("models/checkpoints/best_hubert_model.pt")

# Hardcoded classes
CLASS_NAMES = ["belly pain", "burping", "cold_hot", "discomfort", "hungry", "lonely", "scared", "tired"]


@asynccontextmanager
async def lifespan(app: FastAPI):
    """
    Lifespan context manager for the FastAPI app.
    Handles startup (before yield) and shutdown (after yield) logic.
    """
    global model_instance

    # --- STARTUP LOGIC ---
    # 1. Device selection (M1/CUDA/CPU)
    if torch.cuda.is_available():
        device = "cuda"
    elif torch.backends.mps.is_available():
        device = "mps"
    else:
        device = "cpu"

    print(f"Loading model on {device}...")

    try:
        # Initialize the architecture
        model = HubertClassifier(
            model_name="ntu-spml/distilhubert",
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

    # Yield control back to FastAPI to handle requests
    yield

    # --- SHUTDOWN LOGIC (Optional) ---
    # Example: Clean up GPU memory if needed
    # if model_instance:
    #     del model_instance
    #     torch.cuda.empty_cache()
    print("Shutting down application...")


# Initialize FastAPI with the lifespan handler
app = FastAPI(title="Infant Cry Audio Classifier", lifespan=lifespan)


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

        # 2. Run Inference
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
    uvicorn.run(app, host="0.0.0.0", port=8000)
