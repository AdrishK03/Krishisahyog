import io
import os
import numpy as np
from PIL import Image
from app.services.model_manager import call_hf_inference_api
from .disease_classes import DISEASE_CLASSES, DISEASE_DISPLAY
from .treatments import get_treatment

def _extract_top_prediction(preds):
    if isinstance(preds, dict):
        if "error" in preds:
            raise RuntimeError(preds.get("error", "Hugging Face API returned an error"))
        return preds
    if isinstance(preds, list) and len(preds) > 0:
        return preds[0]
    raise RuntimeError("Unexpected Hugging Face inference response format")


def _display_label(raw_label: str) -> str:
    return DISEASE_DISPLAY.get(raw_label, raw_label)


def predict_plant_disease(image_bytes: bytes, filename: str = "", tta: bool = False) -> dict:
    """
    Two-step API Pipeline:
    1. Identify Plant Type
    2. Identify Disease for that specific plant
    """
    try:
        # STEP 1: Identify Plant
        plant_preds = call_hf_inference_api(image_bytes, "primary")
        top_plant = _extract_top_prediction(plant_preds)

        plant_name = str(top_plant.get('label', 'Unknown')).strip()
        plant_conf = float(top_plant.get('score', 0.0))

        # STEP 2: Identify Disease
        disease_name = "Healthy / Unknown"
        disease_conf = 0.0
        severity = "medium"
        treatment_info = None

        # Only proceed if we are confident about the plant type
        if plant_conf > 0.40 and plant_name.lower() != "unknown":
            disease_preds = call_hf_inference_api(image_bytes, plant_name.lower())
            top_disease = _extract_top_prediction(disease_preds)

            disease_name = str(top_disease.get('label', disease_name)).strip()
            disease_conf = float(top_disease.get('score', 0.0))
            disease_name = _display_label(disease_name)

            treatment_info = get_treatment(plant_name, disease_name)
            if treatment_info and treatment_info.get("severity"):
                severity = treatment_info.get("severity")

        return {
            "status": "success",
            "prediction": disease_name,
            "disease": disease_name,
            "confidence": round(disease_conf * 100, 2),
            "severity": severity,
            "model_used": "HF_Inference_API",
            "plant": {
                "name": plant_name,
                "confidence": round(plant_conf * 100, 2)
            },
            "treatment": treatment_info.get("treatment", []) if treatment_info else ["No treatment data found."],
            "prevention": treatment_info.get("prevention", []) if treatment_info else ["Keep leaf clean and monitored."]
        }

    except Exception as e:
        return {"error": str(e), "status": "failed", "prediction": "Unknown", "confidence": 0, "disease": "Unknown", "severity": "medium", "model_used": "HF_Inference_API"}
