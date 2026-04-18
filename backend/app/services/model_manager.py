import os
import requests
import joblib
from huggingface_hub import hf_hub_download

# Configuration from Environment Variables
HF_TOKEN = os.getenv("HF_TOKEN")

# This repo will hold your .pkl files (Fertilizer model + Encoders)
HF_ASSET_REPO = os.getenv("HF_REPO_ID") 

# Mapping for the Inference API - Ensure these match your Render Env Var names!
MODEL_ENDPOINTS = {
    "primary": os.getenv("HF_REPO_ID_PRIMARY"),
    "corn": os.getenv("HF_REPO_ID_CORN"),
    "rice": os.getenv("HF_REPO_ID_RICE"),
    "potato": os.getenv("HF_REPO_ID_POTATO"),
    "sugarcane": os.getenv("HF_REPO_ID_SUGARCANE"),
    "tomato": os.getenv("HF_REPO_ID_TOMATO")
}

_pkl_cache = {}

def call_hf_inference_api(image_bytes: bytes, model_key: str):
    """Sends image to HF Serverless Inference API."""
    repo_id = MODEL_ENDPOINTS.get(model_key)
    
    if not repo_id:
        raise RuntimeError(f"Endpoint for {model_key} not configured in Env Vars.")
        
    api_url = f"https://api-inference.huggingface.co/models/{repo_id}"
    headers = {
        "Authorization": f"Bearer {HF_TOKEN}",
        "Content-Type": "application/octet-stream",
        "Accept": "application/json",
    }
    
    # We use a timeout because the Free API can sometimes hang
    response = requests.post(api_url, headers=headers, data=image_bytes, timeout=120)
    
    if response.status_code == 200:
        return response.json()
    elif response.status_code == 503:
        # This is common! It means the model is "sleeping" and needs to wake up.
        raise RuntimeError("Model is currently loading on Hugging Face. Try again in 20 seconds.")
    else:
        raise RuntimeError(f"HF API Error: {response.status_code} - {response.text}")

def get_soil_model():
    return _load_pkl("fertilizer_model.pkl")

def get_soil_encoder(name: str):
    # This handles soil_encoder, crop_encoder, etc.
    return _load_pkl(f"{name}.pkl")

def _load_pkl(filename: str):
    """Downloads tiny .pkl files to the Render instance."""
    if filename in _pkl_cache:
        return _pkl_cache[filename]
    
    if not HF_ASSET_REPO:
        raise RuntimeError("HF_REPO_ID (for soil assets) not configured.")

    path = hf_hub_download(
        repo_id=HF_ASSET_REPO,
        filename=filename,
        token=HF_TOKEN
    )
    obj = joblib.load(path)
    _pkl_cache[filename] = obj
    return obj

# Mock functions to keep main.py/plant_predictor.py structure intact
def get_plant_primary_model():
    return "HF_API_MODE"

def get_plant_disease_model(plant_name: str):
    return "HF_API_MODE"
