import io
import os
import numpy as np
from PIL import Image
from app.services.model_manager import call_hf_inference_api
from .disease_classes import DISEASE_CLASSES, DISEASE_DISPLAY
from .treatments import get_treatment

def predict_plant_disease(image_bytes: bytes, filename: str = "", tta: bool = False) -> dict:
    """
    Two-step API Pipeline:
    1. Identify Plant Type
    2. Identify Disease for that specific plant
    """
    try:
        # STEP 1: Identify Plant
        # API returns: [{"label": "Tomato", "score": 0.98}, ...]
        plant_preds = call_hf_inference_api(image_bytes, "primary")
        
        if not plant_preds or "error" in plant_preds:
            raise RuntimeError(plant_preds.get("error", "Plant ID API failed"))

        top_plant = plant_preds
        plant_name = top_plant['label']  # e.g., "Tomato"
        plant_conf = top_plant['score']

        # STEP 2: Identify Disease
        disease_name = "Healthy / Unknown"
        disease_conf = 0.0
        treatment_info = None
        
        # Only proceed if we are confident about the plant type
        if plant_conf > 0.40 and plant_name.lower() != "unknown":
            disease_preds = call_hf_inference_api(image_bytes, plant_name.lower())
            
            if disease_preds and not isinstance(disease_preds, dict):
                top_disease = disease_preds
                disease_name = top_disease['label']
                disease_conf = top_disease['score']
                
                # STEP 3: Get Treatment from your treatments.py
                treatment_info = get_treatment(plant_name, disease_name)

        return {
            "prediction": disease_name,
            "confidence": round(disease_conf * 100, 2),
            "model_used": "HF_Inference_API",
            "plant": {
                "name": plant_name,
                "confidence": round(plant_conf * 100, 2)
            },
            "treatment": treatment_info.get("treatment", []) if treatment_info else ["No treatment data found."],
            "prevention": treatment_info.get("prevention", []) if treatment_info else ["Keep leaf clean and monitored."]
        }

    except Exception as e:
        return {"error": str(e), "status": "failed"}
