"""
Plant disease prediction module.
Loads model from backend/models/plant_disease_model.pkl or uses dummy logic.
"""
import os
import pickle
from pathlib import Path

# Path to model file (relative to backend/)
MODEL_PATH = Path(__file__).resolve().parent.parent / "models" / "plant_disease_model.pkl"

_model = None


def _load_model():
    """Attempt to load the trained model. Returns model or None."""
    global _model
    if _model is not None:
        return _model
    if not MODEL_PATH.exists():
        return None
    try:
        with open(MODEL_PATH, "rb") as f:
            _model = pickle.load(f)
        return _model
    except Exception:
        return None


def predict_plant_disease(image_bytes: bytes, filename: str = "") -> dict:
    """
    Predict plant disease from image.
    Returns: {prediction, confidence, model_used: "real"|"dummy"}
    """
    model = _load_model()
    if model is not None:
        try:
            # Expect model to have predict(image) or similar
            # Common sklearn-style: model.predict(features)
            # For image models, we'd typically use a preprocessor
            # Generic handling: if model has predict_proba, use it
            import numpy as np
            from PIL import Image
            import io

            img = Image.open(io.BytesIO(image_bytes))
            img = img.resize((224, 224))  # Common input size
            arr = np.array(img)
            if len(arr.shape) == 2:
                arr = np.stack([arr] * 3, axis=-1)
            arr = arr.reshape(1, -1)  # Flatten for simple sklearn models

            if hasattr(model, "predict_proba"):
                probs = model.predict_proba(arr)[0]
                idx = probs.argmax()
                confidence = float(probs[idx])
                labels = getattr(model, "classes_", ["Healthy", "Disease"])
                prediction = labels[idx] if idx < len(labels) else "Unknown"
            else:
                pred = model.predict(arr)[0]
                prediction = str(pred)
                confidence = 0.85

            return {
                "prediction": prediction,
                "confidence": round(confidence * 100, 2),
                "model_used": "real",
            }
        except Exception:
            pass  # Fall through to dummy

    # Dummy fallback
    return {
        "prediction": "Model not loaded. Returning demo diagnosis.",
        "confidence": 0.0,
        "model_used": "dummy",
    }
