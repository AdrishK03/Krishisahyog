"""Fertilizer recommendation from soil sensor features + soil/crop type (joblib models)."""
from __future__ import annotations

from pathlib import Path

import joblib
import numpy as np
from pydantic import BaseModel

_MODEL_DIR = Path(__file__).resolve().parent / "model"

model = None
le_target = None
le_crop = None
le_soil = None
_LOAD_ERROR: str | None = None

try:
    model = joblib.load(_MODEL_DIR / "fertilizer_model.pkl")
    le_target = joblib.load(_MODEL_DIR / "target_encoder.pkl")
    le_crop = joblib.load(_MODEL_DIR / "crop_encoder.pkl")
    le_soil = joblib.load(_MODEL_DIR / "soil_encoder.pkl")
except Exception as e:  # pragma: no cover - depends on deployed artifacts
    _LOAD_ERROR = str(e)


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
    if model is None or le_target is None or le_crop is None or le_soil is None:
        return {
            "status": "error",
            "message": _LOAD_ERROR or "Fertilizer models are not loaded.",
        }
    try:
        soil_idx = le_soil.transform([data.Soil_Type])[0]
        crop_idx = le_crop.transform([data.Crop_Type])[0]

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
        fertilizer = le_target.inverse_transform(pred)[0]

        return {"status": "success", "fertilizer": str(fertilizer)}
    except Exception as e:
        return {"status": "error", "message": str(e)}
