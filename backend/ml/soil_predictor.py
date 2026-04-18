"""Fertilizer recommendation from soil sensor features using cached HF assets."""
from __future__ import annotations
import os
import joblib
import numpy as np
from pydantic import BaseModel
from huggingface_hub import hf_hub_download

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
        path = hf_hub_download(repo_id=repo_id, filename=filename, token=token)
        _ASSETS[filename] = joblib.load(path)
    return _ASSETS[filename]

def predict_soil_fertilizer(data: InputData) -> dict:
    try:
        # Load necessary encoders and model
        model = get_asset("fertilizer_model.pkl")
        soil_enc = get_asset("soil_encoder.pkl")
        crop_enc = get_asset("crop_encoder.pkl")
        target_enc = get_asset("target_encoder.pkl")
        
        # Transform categorical inputs
        soil_idx = soil_enc.transform([data.Soil_Type])
        crop_idx = crop_enc.transform([data.Crop_Type])

        n, p, k = data.Nitrogen, data.Phosphorous, data.Potassium
        features = np.array([[
            data.Temperature, data.Humidity, data.Moisture,
            float(soil_idx), float(crop_idx), n, k, p,
            n + p + k, n / (p + 1.0), k / (p + 1.0),
            data.Temperature * data.Moisture,
        ]])

        pred = model.predict(features)
        fertilizer = target_enc.inverse_transform(pred)
        return {"status": "success", "fertilizer": str(fertilizer)}
    except Exception as e:
        return {"status": "error", "message": str(e)}
