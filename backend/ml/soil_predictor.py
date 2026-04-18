"""Fertilizer recommendation from soil sensor features using cached HF assets."""
from __future__ import annotations
import logging
import os
import joblib
import numpy as np
from pydantic import BaseModel
from huggingface_hub import hf_hub_download

logger = logging.getLogger(__name__)

# Import from the correct path as per your structure
from app.services.model_manager import get_soil_model, get_soil_encoder

class InputData(BaseModel):
    Temperature: float
    Humidity: float
    Moisture: float
    Soil_Type: str
    Crop_Type: str
    Nitrogen: float
    Potassium: float
    Phosphorous: float

# Global cache to avoid re-loading on every request
_ASSETS = {}

def get_asset(filename: str):
    """Downloads and caches small .pkl assets from HF."""
    if filename not in _ASSETS:
        repo_id = os.getenv("HF_REPO_ID")
        token = os.getenv("HF_TOKEN")
        if not repo_id:
            raise RuntimeError("HF_REPO_ID is not configured for soil asset downloads.")
        if not token:
            raise RuntimeError("HF_TOKEN is not configured for HF asset downloads.")

        try:
            path = hf_hub_download(repo_id=repo_id, filename=filename, token=token)
        except Exception as exc:
            logger.error("Failed to download HF asset %s from %s", filename, repo_id, exc_info=exc)
            raise RuntimeError(f"Unable to download asset '{filename}' from Hugging Face: {exc}") from exc

        _ASSETS[filename] = joblib.load(path)
    return _ASSETS[filename]

def predict_soil_fertilizer(data: InputData) -> dict:
    try:
        # 1. Load assets
        model = get_asset("fertilizer_model.pkl")
        soil_enc = get_asset("soil_encoder.pkl")
        crop_enc = get_asset("crop_encoder.pkl")
        target_enc = get_asset("target_encoder.pkl")
        
        # 2. Transform categorical inputs
        soil_idx_raw = soil_enc.transform([data.Soil_Type])
        crop_idx_raw = crop_enc.transform([data.Crop_Type])

        soil_idx = int(np.asarray(soil_idx_raw).ravel()[0])
        crop_idx = int(np.asarray(crop_idx_raw).ravel()[0])

        n, p, k = data.Nitrogen, data.Phosphorous, data.Potassium
        
        # 3. Feature Engineering (Must match your training logic exactly)
        features = np.array([[
            float(data.Temperature),
            float(data.Humidity),
            float(data.Moisture),
            float(soil_idx),
            float(crop_idx),
            float(n),
            float(k),
            float(p),
            float(n + p + k),
            float(n / (p + 1.0)),
            float(k / (p + 1.0)),
            float(data.Temperature * data.Moisture),
        ]], dtype=float)

        # 4. Predict
        pred = model.predict(features)
        
        fertilizer_raw = target_enc.inverse_transform(pred)
        fertilizer = str(np.asarray(fertilizer_raw).ravel()[0])

        return {"status": "success", "fertilizer": fertilizer}
    except Exception as e:
        logger.error("Soil fertilizer prediction failed: %s", e, exc_info=e)
        return {"status": "error", "message": str(e)}
