"""Fertilizer recommendation from soil sensor features using cached HF assets."""
from __future__ import annotations
import os
import joblib
import numpy as np
from pydantic import BaseModel
from huggingface_hub import hf_hub_download

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
        # Ensure your HF_REPO_ID points to the repo with .pkl files
        path = hf_hub_download(repo_id=repo_id, filename=filename, token=token)
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
        # FIX: Added to extract the scalar from the returned array
        soil_idx = soil_enc.transform([data.Soil_Type])
        crop_idx = crop_enc.transform([data.Crop_Type])

        n, p, k = data.Nitrogen, data.Phosphorous, data.Potassium
        
        # 3. Feature Engineering (Must match your training logic exactly)
        features = np.array([[
            data.Temperature, 
            data.Humidity, 
            data.Moisture,
            float(soil_idx), 
            float(crop_idx), 
            n, k, p,
            n + p + k, 
            n / (p + 1.0), 
            k / (p + 1.0),
            data.Temperature * data.Moisture,
        ]])

        # 4. Predict
        pred = model.predict(features)
        
        # FIX: Added to get the first prediction result string
        fertilizer = target_enc.inverse_transform(pred)
        
        return {"status": "success", "fertilizer": str(fertilizer)}
    except Exception as e:
        return {"status": "error", "message": str(e)}
