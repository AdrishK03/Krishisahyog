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
            raise RuntimeError(preds.get("error", "HF API Error"))
        return preds
    if isinstance(preds, list) and len(preds) > 0:
        return preds
    raise RuntimeError("Unexpected HF response format")

def predict_plant_disease(image_bytes: bytes, filename: str = "", tta: bool = False) -> dict:
    try:
        # STEP 1: Identify Plant
        plant_preds = call_hf_inference_api(image_bytes, "primary")
        top_plant = _extract_top_prediction(plant_preds)

        # HF returns labels like "Rice", "Tomato". We normalize to Capital Case.
        plant_name = str(top_plant.get('label', 'Unknown')).strip().capitalize()
        plant_conf = float(top_plant.get('score', 0.0))

        # STEP 2: Identify Disease
        disease_name = "Healthy"
        disease_conf = 0.0
        severity = "low"
        treatment_info = None

        supported_plants = ["rice", "potato", "corn", "sugarcane", "tomato"]
        
        if plant_conf > 0.40 and plant_name.lower() in supported_plants:
            # Second API call to the specific crop model
            disease_preds = call_hf_inference_api(image_bytes, plant_name.lower())
            top_disease = _extract_top_prediction(disease_preds)

            # This 'raw_label' now matches your config.json (e.g., "Potato___Early_blight")
            raw_label = str(top_disease.get('label', 'Healthy')).strip()
            disease_conf = float(top_disease.get('score', 0.0))
            
            # Get pretty name for UI (e.g., "Early Blight")
            disease_name = DISEASE_DISPLAY.get(raw_label, raw_label)

            # STEP 3: Get Treatment (Key format: "Plant|Raw_Label")
            treatment_info = get_treatment(plant_name, raw_label)
            
            # Backup: if lookup fails, try with the display name
            if not treatment_info:
                treatment_info = get_treatment(plant_name, disease_name)

            if treatment_info:
                severity = treatment_info.get("severity", "medium")

        return {
            "status": "success",
            "prediction": disease_name,
            "disease": disease_name,
            "confidence": round(disease_conf * 100, 2),
            "severity": severity,
            "model_used": "HF_Serverless_Inference",
            "plant": {
                "name": plant_name,
                "confidence": round(plant_conf * 100, 2)
            },
            "treatment": treatment_info.get("fungicide", []) + treatment_info.get("cultural", []) if treatment_info else ["No treatment data found."],
            "prevention": treatment_info.get("organic", []) if treatment_info else ["Monitor crop condition."]
        }

    except Exception as e:
        return {
            "error": str(e), 
            "status": "failed", 
            "prediction": "Unknown", 
            "confidence": 0, 
            "disease": "Unknown",
            "severity": "medium"
        }
