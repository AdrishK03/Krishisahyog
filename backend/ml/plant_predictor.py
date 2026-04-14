"""
PlantSense API v7 — FastAPI Backend
Pipeline:
  1) Plant ID  → plant_model.keras    (TF Keras)
  2) Disease   → potato/tomato/…     (TF Keras)
  3) Treatment → treatments.py

Run:
  uvicorn main:app --port 8000
"""

import os
from fastapi import FastAPI, File, UploadFile, HTTPException
from fastapi.middleware.cors import CORSMiddleware
from fastapi.staticfiles import StaticFiles
from fastapi.responses import FileResponse
import numpy as np
from PIL import Image
import io, time, logging, base64
from pathlib import Path
from typing import Optional

from app.services.model_manager import (
    get_plant_primary_model,
    get_plant_disease_model,
)

logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

# ── Local modules ─────────────────────────────────────────────────────────────
try:
    from .disease_classes import DISEASE_CLASSES, DISEASE_DISPLAY
    from .treatments import get_treatment
except ImportError:
    from disease_classes import DISEASE_CLASSES, DISEASE_DISPLAY
    from treatments import get_treatment

# ─────────────────────────────────────────────────────────────────────────────
# CONFIG
# ─────────────────────────────────────────────────────────────────────────────
BASE_DIR = Path(__file__).resolve().parents[1]

PLANT_THRESHOLD   = 0.55   # below → Unknown, stop pipeline
DISEASE_THRESHOLD = 0.30   # below → disease uncertain

IMG_SIZE      = 224
ALLOWED_TYPES = {"image/jpeg", "image/png", "image/webp", "image/bmp"}
MAX_FILE_SIZE = 10 * 1024 * 1024   # 10 MB

# ⚠️  Must match EXACT class order used during training of plant_model.keras
PLANT_CLASSES    = ["Unknown", "Sugarcane", "Tomato", "Rice", "Potato", "Corn"]
SUPPORTED_PLANTS = {"Sugarcane", "Potato", "Rice", "Tomato", "Corn"}

CLASS_INFO = {
    "Unknown":   {"emoji": "❓", "color": "#6b7280"},
    "Sugarcane": {"emoji": "🌾", "color": "#f59e0b"},
    "Tomato":    {"emoji": "🍅", "color": "#ef4444"},
    "Rice":      {"emoji": "🌾", "color": "#84cc16"},
    "Potato":    {"emoji": "🥔", "color": "#d97706"},
    "Corn":      {"emoji": "🌽", "color": "#eab308"},
}

# ─────────────────────────────────────────────────────────────────────────────
# TENSORFLOW
# ─────────────────────────────────────────────────────────────────────────────
try:
    import tensorflow as tf
    tf.config.set_visible_devices([], "GPU")   # CPU only on local machine
    tf.get_logger().setLevel("ERROR")
    TF_AVAILABLE = True
    logger.info(f"TensorFlow {tf.__version__} loaded (CPU mode)")
except ImportError:
    TF_AVAILABLE = False
    logger.error("TensorFlow not installed. Run: pip install tensorflow-cpu")

# ─────────────────────────────────────────────────────────────────────────────
# PLANT MODEL — Keras
# ─────────────────────────────────────────────────────────────────────────────
plant_model: Optional[object] = None
plant_input_size: tuple = (IMG_SIZE, IMG_SIZE)


def load_plant_model():
    global plant_input_size

    if not TF_AVAILABLE:
        logger.error("Cannot load plant model — TensorFlow not installed.")
        return None
    try:
        m = get_plant_primary_model()

        # Auto-detect input size from model
        shape = m.input_shape
        if isinstance(shape, list):
            shape = shape[0]
        h = int(shape[1]) if shape[1] else IMG_SIZE
        w = int(shape[2]) if shape[2] else IMG_SIZE
        plant_input_size = (h, w)

        n_out = m.output_shape[-1]
        n_cls = len(PLANT_CLASSES)
        if n_out != n_cls:
            logger.warning(
                f"⚠️  Plant model output={n_out} but PLANT_CLASSES has {n_cls} — "
                f"check PLANT_CLASSES order in main.py!"
            )
        else:
            logger.info(
                f"✅ Plant model loaded | input=({h},{w}) | classes={n_cls}"
            )
        return m

    except Exception as e:
        logger.error(f"❌ Plant model load failed: {e}", exc_info=True)
        return None


disease_models: dict = {}
disease_input_sizes: dict = {}


def _get_disease_filename(plant: str) -> str:
    raw = os.getenv("PLANT_MODEL_FILENAME", "")
    parts = [p.strip() for p in raw.replace(",", "\\").split("\\") if p.strip()]
    for name in parts:
        if plant.lower() in name.lower():
            return name
    return f"{plant.lower()}.keras"


def get_disease_model(plant: str):
    if not TF_AVAILABLE:
        return None
    if plant in disease_models:
        return disease_models[plant]
    try:
        model = get_plant_disease_model(plant)
        disease_models[plant] = model
        shape = model.input_shape
        if isinstance(shape, list):
            shape = shape[0]
        h = int(shape[1]) if shape[1] else 224
        w = int(shape[2]) if shape[2] else 224
        disease_input_sizes[plant] = (h, w)
        return model
    except Exception as e:
        logger.error(f"❌ Failed to load {plant} disease model: {e}", exc_info=True)
        return None

# ─────────────────────────────────────────────────────────────────────────────
# IMAGE PREPROCESSING
# ─────────────────────────────────────────────────────────────────────────────
def prepare_image(img_array: np.ndarray, h: int, w: int) -> np.ndarray:
    """
    Resize to (h, w), keep pixel values in [0, 255] float32.
    Models have include_preprocessing=True so they handle
    normalization internally — do NOT divide by 255 here.
    Returns shape (1, h, w, 3).
    """
    img = Image.fromarray(img_array).resize((w, h))
    arr = np.array(img, dtype=np.float32)
    return np.expand_dims(arr, axis=0)


# ─────────────────────────────────────────────────────────────────────────────
# STEP 1 — PLANT IDENTIFICATION
# ─────────────────────────────────────────────────────────────────────────────
def identify_plant(img_array: np.ndarray) -> dict:
    global plant_model
    if plant_model is None:
        plant_model = load_plant_model()
    if plant_model is None:
        raise RuntimeError("Plant model not loaded.")

    h, w = plant_input_size
    inp  = prepare_image(img_array, h, w)

    t0    = time.perf_counter()
    preds = plant_model.predict(inp, verbose=0)[0]   # shape: (num_classes,)
    elapsed = (time.perf_counter() - t0) * 1000

    idx        = int(np.argmax(preds))
    conf       = float(preds[idx])
    pred_plant = PLANT_CLASSES[idx]
    accepted   = (pred_plant in SUPPORTED_PLANTS) and (conf >= PLANT_THRESHOLD)

    return {
        "plant":           pred_plant if accepted else "Unknown",
        "predicted_class": pred_plant,
        "confidence":      conf,
        "all_probs":       {c: round(float(p), 6) for c, p in zip(PLANT_CLASSES, preds)},
        "elapsed_ms":      round(elapsed, 1),
        "accepted":        accepted,
    }


# ─────────────────────────────────────────────────────────────────────────────
# STEP 2 — DISEASE DETECTION
# ─────────────────────────────────────────────────────────────────────────────
def detect_disease(img_array: np.ndarray, plant: str) -> dict:
    model = get_disease_model(plant)
    if model is None:
        return {
            "disease":    None,
            "confidence": None,
            "accepted":   False,
            "reason":     f"Disease model not available for {plant}",
        }

    disease_class_list = DISEASE_CLASSES[plant]
    h, w = disease_input_sizes.get(plant, (224, 224))
    inp  = prepare_image(img_array, h, w)

    t0      = time.perf_counter()
    preds = model.predict(inp, verbose=0)[0]
    elapsed = (time.perf_counter() - t0) * 1000

    idx      = int(np.argmax(preds))
    conf     = float(preds[idx])
    raw_name = disease_class_list[idx] if idx < len(disease_class_list) else f"Class_{idx}"
    display  = DISEASE_DISPLAY.get(raw_name, raw_name)

    all_probs = {
        DISEASE_DISPLAY.get(disease_class_list[i], disease_class_list[i]): round(float(p), 6)
        for i, p in enumerate(preds)
        if i < len(disease_class_list)
    }

    return {
        "disease":    display,
        "raw_class":  raw_name,
        "confidence": conf,
        "all_probs":  all_probs,
        "elapsed_ms": round(elapsed, 1),
        "accepted":   conf >= DISEASE_THRESHOLD,
        "reason":     None if conf >= DISEASE_THRESHOLD else f"Low confidence ({conf*100:.1f}%)",
    }


# ─────────────────────────────────────────────────────────────────────────────
# FULL PIPELINE
# ─────────────────────────────────────────────────────────────────────────────
def run_pipeline(img_array: np.ndarray) -> dict:

    # Step 1 — Plant ID
    plant_result = identify_plant(img_array)

    if not plant_result["accepted"]:
        return {
            "stage": "plant_identification",
            "plant": {
                "name":            "Unknown",
                "predicted_class": plant_result["predicted_class"],
                "confidence":      plant_result["confidence"],
                "all_probs":       plant_result["all_probs"],
                "emoji":           "❓",
                "accepted":        False,
            },
            "disease":   None,
            "treatment": None,
            "total_ms":  plant_result["elapsed_ms"],
            "message":   f"Plant not confidently identified (threshold {PLANT_THRESHOLD:.2f}).",
        }

    plant_name = plant_result["plant"]
    info       = CLASS_INFO.get(plant_name, {})

    # Step 2 — Disease
    disease_result = detect_disease(img_array, plant_name)
    total_ms = plant_result["elapsed_ms"] + (disease_result.get("elapsed_ms") or 0)

    # Step 3 — Treatment
    treatment = None
    if disease_result["accepted"] and disease_result.get("raw_class"):
        treatment = get_treatment(plant_name, disease_result["raw_class"])
        if treatment is None:
            logger.warning(
                f"No treatment entry: '{plant_name}|{disease_result['raw_class']}' "
                f"— check treatments.py"
            )

    return {
        "stage": "complete",
        "plant": {
            "name":            plant_name,
            "predicted_class": plant_result["predicted_class"],
            "confidence":      plant_result["confidence"],
            "all_probs":       plant_result["all_probs"],
            "emoji":           info.get("emoji", "🌱"),
            "accepted":        True,
        },
        "disease": {
            "name":       disease_result["disease"],
            "raw_class":  disease_result.get("raw_class"),
            "confidence": disease_result["confidence"],
            "all_probs":  disease_result.get("all_probs", {}),
            "accepted":   disease_result["accepted"],
            "reason":     disease_result.get("reason"),
        } if disease_result.get("disease") else None,
        "treatment": treatment,
        "total_ms":  round(total_ms, 1),
        "message":   None,
    }


# ─────────────────────────────────────────────────────────────────────────────
# HELPER FUNCTIONS
# ─────────────────────────────────────────────────────────────────────────────
def _normalize_severity(raw: Optional[str]) -> str:
    value = (raw or "").strip().lower()
    if "severe" in value or "high" in value:    return "high"
    if "moderate" in value or "medium" in value: return "medium"
    if "none" in value or "healthy" in value or "low" in value: return "low"
    return "medium"


def _build_prevention(treatment: Optional[dict]) -> list[str]:
    if not treatment: return []
    prevention: list[str] = []
    cultural = treatment.get("cultural")
    if isinstance(cultural, list):
        prevention.extend([str(i) for i in cultural])
    elif cultural:
        prevention.append(str(cultural))
    organic = treatment.get("organic")
    if organic:
        prevention.append(f"Organic: {organic}")
    return prevention


def _build_treatment_list(treatment: Optional[dict]) -> list[str]:
    if not treatment: return []
    lines: list[str] = []
    if treatment.get("cause"):      lines.append(f"Cause: {treatment['cause']}")
    if treatment.get("fertilizer"): lines.append(f"Fertilizer: {treatment['fertilizer']}")
    fungicide = treatment.get("fungicide")
    if isinstance(fungicide, list):
        lines.extend([f"Fungicide: {item}" for item in fungicide if item])
    elif fungicide:
        lines.append(f"Fungicide: {fungicide}")
    return lines


def predict_plant_disease(image_bytes: bytes, filename: str = "", tta: bool = False) -> dict:
    """
    Wrapper function for compatibility with main.py.
    Accepts raw image bytes and returns prediction results.
    """
    if not image_bytes:
        raise RuntimeError("Empty image payload.")

    img       = Image.open(io.BytesIO(image_bytes)).convert("RGB")
    img_array = np.array(img)
    pipeline  = run_pipeline(img_array)

    stage        = pipeline.get("stage")
    disease_info = pipeline.get("disease")
    treatment_info = pipeline.get("treatment")

    disease_name = "Unknown"
    disease_conf = 0.0
    if disease_info:
        disease_name = disease_info.get("name") or "Unknown"
        disease_conf = float(disease_info.get("confidence") or 0.0)

    result = {
        "prediction": disease_name,
        "confidence": round(disease_conf * 100, 2),
        "model_used": "real",
        "disease":    disease_name,
        "severity":   _normalize_severity((treatment_info or {}).get("severity")) if disease_info else "low",
        "treatment":  _build_treatment_list(treatment_info),
        "prevention": _build_prevention(treatment_info),
        "plant":      pipeline.get("plant", {}),
        "pipeline":   pipeline,
        "filename":   filename,
    }

    if not result["treatment"] and disease_name == "Unknown":
        if stage == "plant_identification":
            result["treatment"]  = ["Plant could not be confidently identified. Upload a clearer leaf image."]
            result["prevention"] = ["Capture image in daylight, focus on affected area, and avoid blur."]
        else:
            result["treatment"]  = ["Disease confidence is low. Please re-upload a clearer image of the affected area."]
            result["prevention"] = ["Ensure good lighting, keep the leaf fully visible, and focus on symptoms."]

    return result





# ─────────────────────────────────────────────────────────────────────────────
# FASTAPI APP
# ─────────────────────────────────────────────────────────────────────────────
app = FastAPI(
    title="🌿 PlantSense API v7",
    description="Plant ID (Keras) → Disease Detection (Keras) → Treatment Advice",
    version="7.0.0",
)

app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_methods=["*"],
    allow_headers=["*"],
)

static_dir = Path("static")
if static_dir.exists():
    app.mount("/static", StaticFiles(directory="static"), name="static")


@app.get("/")
async def root():
    if Path("static/index.html").exists():
        return FileResponse("static/index.html")
    return {"message": "PlantSense API v7 is running", "docs": "/docs"}


@app.get("/health")
async def health():
    return {
        "status":                "ok",
        "tensorflow":            TF_AVAILABLE,
        "plant_model_loaded":    plant_model is not None,
        "plant_input_size":      plant_input_size,
        "plant_classes":         PLANT_CLASSES,
        "supported_plants":      list(SUPPORTED_PLANTS),
        "disease_models_loaded": list(disease_models.keys()),
        "disease_input_sizes":   disease_input_sizes,
        "plant_threshold":       PLANT_THRESHOLD,
        "disease_threshold":     DISEASE_THRESHOLD,
    }


@app.post("/predict")
async def predict_endpoint(file: UploadFile = File(...)):
    if file.content_type not in ALLOWED_TYPES:
        raise HTTPException(
            status_code=400,
            detail=f"Unsupported type: {file.content_type}. Use JPEG / PNG / WebP / BMP."
        )

    raw = await file.read()
    if len(raw) > MAX_FILE_SIZE:
        raise HTTPException(status_code=413, detail="File too large. Max 10 MB.")
    try:
        img       = Image.open(io.BytesIO(raw)).convert("RGB")
        img_array = np.array(img)
        result    = run_pipeline(img_array)

        thumb = img.resize((224, 224))
        buf   = io.BytesIO()
        thumb.save(buf, format="JPEG", quality=80)
        result["thumbnail"]     = "data:image/jpeg;base64," + base64.b64encode(buf.getvalue()).decode()
        result["original_size"] = {"w": img.width, "h": img.height}
        result["filename"]      = file.filename
        return result

    except RuntimeError as e:
        raise HTTPException(status_code=503, detail=str(e))
    except Exception as e:
        logger.error(f"Prediction error: {e}", exc_info=True)
        raise HTTPException(status_code=500, detail=f"Prediction failed: {str(e)}")


@app.get("/classes")
async def get_classes():
    return {
        "plants":           PLANT_CLASSES,
        "supported_plants": list(SUPPORTED_PLANTS),
        "disease_classes":  DISEASE_CLASSES,
        "disease_display":  DISEASE_DISPLAY,
        "plant_threshold":  PLANT_THRESHOLD,
        "disease_threshold": DISEASE_THRESHOLD,
    }


@app.post("/reload-models")
async def reload_models():
    """Hot-reload all models without restarting the server."""
    global plant_model, plant_input_size
    plant_model = None
    plant_input_size = (IMG_SIZE, IMG_SIZE)
    disease_models.clear()
    disease_input_sizes.clear()
    return {
        "plant_model":         False,
        "plant_input_size":    plant_input_size,
        "disease_models":      list(disease_models.keys()),
        "disease_input_sizes": disease_input_sizes,
    }