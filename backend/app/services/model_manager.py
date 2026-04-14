import os
import json
import threading
import joblib
import tensorflow as tf
from huggingface_hub import hf_hub_download

# ------- cache stores -------
_keras_cache: dict = {}
_pkl_cache: dict = {}
_json_cache: dict = {}
_lock = threading.Lock()

VALID_PLANT_NAMES = {"corn", "rice", "potato", "sugarcane", "tomato"}


def _download(filename: str) -> str:
    """Download a file from HF Hub to /tmp/hf_models. Returns local path."""
    return hf_hub_download(
        repo_id=os.getenv("HF_REPO_ID"),
        filename=filename,
        token=os.getenv("HF_TOKEN") or None,
        cache_dir="/tmp/hf_models",
    )


def _load_keras(cache_key: str, filename: str):
    if cache_key in _keras_cache:
        return _keras_cache[cache_key]
    with _lock:
        if cache_key in _keras_cache:
            return _keras_cache[cache_key]
        path = _download(filename)
        model = tf.keras.models.load_model(path)
        _keras_cache[cache_key] = model
        print(f"[ModelManager] Loaded keras model: {filename}")
        return model


def _load_pkl(cache_key: str, filename: str):
    if cache_key in _pkl_cache:
        return _pkl_cache[cache_key]
    with _lock:
        if cache_key in _pkl_cache:
            return _pkl_cache[cache_key]
        path = _download(filename)
        obj = joblib.load(path)
        _pkl_cache[cache_key] = obj
        print(f"[ModelManager] Loaded pkl: {filename}")
        return obj


def _load_json(cache_key: str, filename: str) -> dict:
    if cache_key in _json_cache:
        return _json_cache[cache_key]
    with _lock:
        if cache_key in _json_cache:
            return _json_cache[cache_key]
        path = _download(filename)
        with open(path, "r", encoding="utf-8") as f:
            data = json.load(f)
        _json_cache[cache_key] = data
        print(f"[ModelManager] Loaded json: {filename}")
        return data


# ------- public API -------

def get_plant_primary_model():
    filename = os.getenv("PLANT_PRIMARY_MODEL") or os.getenv("PLANT_MODEL_FILENAME", "plant_model.keras")
    return _load_keras("plant_primary", filename)


def get_plant_disease_model(plant_name: str):
    plant_name = plant_name.lower().strip()
    if plant_name not in VALID_PLANT_NAMES:
        raise ValueError(
            f"Unknown plant '{plant_name}'. Valid: {VALID_PLANT_NAMES}"
        )
    filename = f"{plant_name}.keras"
    return _load_keras(f"disease_{plant_name}", filename)


def get_soil_model():
    filename = os.getenv("SOIL_MODEL_FILENAME", "fertilizer_model.pkl")
    return _load_pkl("soil_model", filename)


def get_soil_encoder(name: str):
    valid = {"soil_encoder", "crop_encoder", "target_encoder"}
    if name not in valid:
        raise ValueError(f"Unknown encoder '{name}'. Valid: {valid}")
    return _load_pkl(name, f"{name}.pkl")


def get_soil_config(name: str):
    valid = {"config", "metadata"}
    if name not in valid:
        raise ValueError(f"Unknown config '{name}'. Valid: {valid}")
    return _load_json(name, f"{name}.json")

