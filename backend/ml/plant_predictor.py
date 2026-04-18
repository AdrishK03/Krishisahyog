import io
import time
from PIL import Image
from app.services.model_manager import call_hf_inference_api
from .disease_classes import DISEASE_CLASSES, DISEASE_DISPLAY
from .treatments import get_treatment

def predict_plant_disease(image_bytes: bytes, filename: str = "", tta: bool = False) -> dict:
    # 1. Step 1: Identify Plant
    # Note: API returns a list of dicts: [{'label': 'Tomato', 'score': 0.99}, ...]
    plant_preds = call_hf_inference_api(image_bytes, "primary")
    top_plant = plant_preds
    plant_name = top_plant['label']
    plant_conf = top_plant['score']

    # 2. Step 2: Identify Disease (If recognized)
    disease_name = "Healthy / Unknown"
    disease_conf = 0.0
    treatment = None

    if plant_conf > 0.55 and plant_name.lower() != "unknown":
        disease_preds = call_hf_inference_api(image_bytes, plant_name.lower())
        top_disease = disease_preds
        disease_name = top_disease['label']
        disease_conf = top_disease['score']
        
        # 3. Get Treatment
        treatment = get_treatment(plant_name, disease_name)

    return {
        "prediction": disease_name,
        "confidence": round(disease_conf * 100, 2),
        "plant": {
            "name": plant_name,
            "confidence": round(plant_conf, 4)
        },
        "treatment": treatment,
        "model_used": "huggingface_inference_api"
    }
