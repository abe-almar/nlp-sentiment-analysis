import json
import os

import numpy as np
import tensorflow as tf
from fastapi import FastAPI, HTTPException, status
from loguru import logger
from pydantic import BaseModel
from tensorflow import convert_to_tensor
from tensorflow.keras.models import load_model  # type: ignore

from src.config import idx_to_label
from src.formatting import format_input

# Create FastAPI app
app = FastAPI()

# Load the trained model and preprocessing pipeline
BASE_DIR = os.path.join(os.path.dirname(os.path.dirname(__file__)))
MODEL_PATH = os.path.join(BASE_DIR, "models", "full_model_1D.keras")

try:
    model = load_model(MODEL_PATH)
    logger.info("Keras model loaded successfully.")
except Exception as e:
    logger.error(f"Failed to load Keras model: {e}")
    raise RuntimeError("Model could not be loaded.")


# Define request schema using Pydantic
class SentimentRequest(BaseModel):
    entity: str
    text: str


@app.get("/isalive", status_code=status.HTTP_200_OK)
def isalive():
    """Health check endpoint.

    Returns:
        dict: Health status message.
    """
    return {"health_check": "OK", "message": "NLP sentiment analysis API"}


@app.post("/predict", status_code=status.HTTP_200_OK)
async def predict(request: SentimentRequest):
    """Predict the sentiment of a text about a given entity.

    Args:
        request (SentimentRequest): JSON with 'entity' and 'text'.

    Returns:
        dict: Dictionary containing the predicted class and confidence.

    Raises:
        HTTPException: If there's an error processing the request or making predictions.
    """
    try:
        # Format the input text
        formatted_text = format_input(request.entity, request.text)
        # Convert raw string to tensor
        raw_text_data = convert_to_tensor([formatted_text], dtype=tf.string)
        # Predict using the model
        prediction = model.predict([raw_text_data])
        predicted_class_idx = int(np.argmax(prediction, axis=1)[0])
        predicted_label = idx_to_label[predicted_class_idx]
        confidence = float(np.max(prediction))

        return {
            "entity": request.entity,
            "predicted_class": predicted_label,
            "confidence": confidence,
        }

    except Exception as e:
        logger.error(f"Internal server error: {e}")
        raise HTTPException(
            status_code=status.HTTP_500_INTERNAL_SERVER_ERROR,
            detail="Internal server error.",
        )
