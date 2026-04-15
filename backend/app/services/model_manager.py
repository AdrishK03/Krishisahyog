import os
import json
import threading
from pathlib import Path
import joblib
import tensorflow as tf
from huggingface_hub import hf_hub_download

BASE_DIR = Path(__file__).resolve().parents[2]
LOCAL_MODEL_DIR = BASE_DIR / "models"

# ------- cache stores -------
_keras_cache: dict = {}
_pkl_cache: dict = {}
_json_cache: dict = {}
_lock = threading.Lock()

VALID_PLANT_NAMES = {"corn", "rice", "potato", "sugarcane", "tomato"}


def _local_model_path(filename: str) -> str | None:
    candidate = LOCAL_MODEL_DIR / filename
    if candidate.exists():
        return str(candidate)
    return None


def _download(filename: str) -> str:
    """Download a file from HF Hub or use a local ./models copy when available."""
    local_path = _local_model_path(filename)
    if local_path:
        return local_path

    repo_id = os.getenv("HF_REPO_ID")
    if not repo_id:
        raise RuntimeError(
            f"Model '{filename}' not found locally and HF_REPO_ID is not configured."
        )

    return hf_hub_download(
        repo_id=repo_id,
        filename=filename,
        token=os.getenv("HF_TOKEN") or None,
        cache_dir="/tmp/hf_models",
    )


import keras  # standalone Keras 3, NOT tf.keras

def _load_keras(cache_key: str, filename: str):
    if cache_key in _keras_cache:
        return _keras_cache[cache_key]
    with _lock:
        if cache_key in _keras_cache:
            return _keras_cache[cache_key]
        path = _download(filename)
        try:
            model = keras.models.load_model(path)  # NOT tf.keras
            print(f"[ModelManager] Loaded: {filename} using standalone keras")
        except Exception as first_exc:
            try:
                import tensorflow as tf
                model = tf.keras.models.load_model(path)
                print(f"[ModelManager] Loaded: {filename} using tf.keras fallback")
            except Exception as second_exc:
                raise RuntimeError(
                    f"Failed to load model '{filename}' with keras and tf.keras. "
                    f"standalone error: {first_exc}; tf.keras error: {second_exc}"
                ) from second_exc
        _keras_cache[cache_key] = model
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


def _get_disease_filename(plant_name: str) -> str:
    raw = os.getenv("PLANT_DISEASE_MODELS", "")
    candidates = [p.strip() for p in raw.replace(",", ";").split(";") if p.strip()]
    for candidate in candidates:
        if plant_name.lower() in candidate.lower():
            return candidate
    return f"{plant_name}.keras"


def get_plant_disease_model(plant_name: str):
    plant_name = plant_name.lower().strip()
    if plant_name not in VALID_PLANT_NAMES:
        raise ValueError(
            f"Unknown plant '{plant_name}'. Valid: {VALID_PLANT_NAMES}"
        )
    filename = _get_disease_filename(plant_name)
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

