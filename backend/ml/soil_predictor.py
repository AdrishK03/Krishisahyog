"""Fertilizer recommendation from soil sensor features + soil/crop type (joblib models)."""
from __future__ import annotations

import os

import numpy as np
from pydantic import BaseModel
from app.services.model_manager import (
    get_soil_model,
    get_soil_encoder,
    get_soil_config,
)


class InputData(BaseModel):
    Temperature: float
    Humidity: float
    Moisture: float
    Soil_Type: str
    Crop_Type: str
    Nitrogen: float
    Potassium: float
    Phosphorous: float


def predict_soil_fertilizer(data: InputData) -> dict:
    """
    Returns {"status": "success", "fertilizer": str} or
    {"status": "error", "message": str}.
    """
    try:
        model = get_soil_model()
        soil_enc = get_soil_encoder("soil_encoder")
        crop_enc = get_soil_encoder("crop_encoder")
        target_enc = get_soil_encoder("target_encoder")
        config = get_soil_config("config")
        metadata = get_soil_config("metadata")
    except Exception as e:
        return {
            "status": "error",
            "message": f"Model loading failed: {str(e)}",
        }

    try:
        soil_idx = soil_enc.transform([data.Soil_Type])[0]
        crop_idx = crop_enc.transform([data.Crop_Type])[0]

        n, p, k = data.Nitrogen, data.Phosphorous, data.Potassium

        features = np.array(
            [
                [
                    data.Temperature,
                    data.Humidity,
                    data.Moisture,
                    float(soil_idx),
                    float(crop_idx),
                    n,
                    k,
                    p,
                    n + p + k,
                    n / (p + 1.0),
                    k / (p + 1.0),
                    data.Temperature * data.Moisture,
                ]
            ]
        )

        pred = model.predict(features)
        fertilizer = target_enc.inverse_transform(pred)[0]

        return {"status": "success", "fertilizer": str(fertilizer)}
    except Exception as e:
        return {"status": "error", "message": str(e)}
