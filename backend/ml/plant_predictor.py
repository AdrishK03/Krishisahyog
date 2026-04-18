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
        
        if isinstance(plant_preds, dict):
            if "error" in plant_preds:
                raise RuntimeError(plant_preds.get("error", "Plant ID API failed"))
            top_plant = plant_preds
        elif isinstance(plant_preds, list) and len(plant_preds) > 0:
            top_plant = plant_preds[0]
        else:
            raise RuntimeError("Unexpected Hugging Face plant prediction format")

        plant_name = str(top_plant.get('label', 'Unknown'))
        plant_conf = float(top_plant.get('score', 0.0))

        # STEP 2: Identify Disease
        disease_name = "Healthy / Unknown"
        disease_conf = 0.0
        treatment_info = None
        
        # Only proceed if we are confident about the plant type
        if plant_conf > 0.40 and plant_name.lower() != "unknown":
            disease_preds = call_hf_inference_api(image_bytes, plant_name.lower())
            
            if isinstance(disease_preds, dict):
                if "error" in disease_preds:
                    raise RuntimeError(disease_preds.get("error", "Disease API failed"))
                top_disease = disease_preds
            elif isinstance(disease_preds, list) and len(disease_preds) > 0:
                top_disease = disease_preds[0]
            else:
                top_disease = None

            if top_disease:
                disease_name = str(top_disease.get('label', 'Healthy / Unknown'))
                disease_conf = float(top_disease.get('score', 0.0))
                
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
