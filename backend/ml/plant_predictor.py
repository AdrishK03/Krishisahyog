"""
PlantSense API v4 — FastAPI Backend
Pipeline:
1) Plant ID using best_model.pth
2) If plant confidence >= 0.52 and plant != Unknown:
      -> run that specific disease model
3) Else:
      -> return Unknown plant and stop

Run:
uvicorn main:app --port 8000
"""

from fastapi import FastAPI, File, UploadFile, HTTPException
from fastapi.middleware.cors import CORSMiddleware
from fastapi.staticfiles import StaticFiles
from fastapi.responses import FileResponse
import torch
import torch.nn as nn
import torch.nn.functional as F
import timm
import numpy as np
from PIL import Image
try:
    import albumentations as A  # type: ignore
    from albumentations.pytorch import ToTensorV2  # type: ignore

    _ALBU_AVAILABLE = True
except Exception as e:  # pragma: no cover - depends on environment
    # If albumentations/albucore are incompatible, keep backend running.
    class _AlbuStub:
        """Minimal stub so module import doesn't fail when albumentations is broken."""

        def __getattr__(self, _name: str):
            return lambda *args, **kwargs: None

        def Compose(self, *args, **kwargs):
            def _call(**t):
                # Keep shape { "image": ... } compatible with albumentations calls.
                return {"image": t.get("image")}

            return _call

    class _ToTensorV2Stub:
        def __call__(self, *args, **kwargs):
            return None

    A = _AlbuStub()  # type: ignore[assignment]
    ToTensorV2 = _ToTensorV2Stub  # type: ignore[assignment]
    _ALBU_AVAILABLE = False
    _ALBU_IMPORT_ERROR = str(e)
import io, time, logging, base64
from pathlib import Path
from typing import Optional

# ── Local modules ─────────────────────────────────────────────────────────────
try:
    from .disease_classes import DISEASE_CLASSES, DISEASE_DISPLAY
    from .treatments import get_treatment
except ImportError:
    from disease_classes import DISEASE_CLASSES, DISEASE_DISPLAY
    from treatments import get_treatment

logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

if not _ALBU_AVAILABLE:
    logger.warning("Albumentations unavailable, using fallback preprocessing: %s", locals().get("_ALBU_IMPORT_ERROR"))

# ─────────────────────────────────────────────────────────────────────────────
# CONFIG
# ─────────────────────────────────────────────────────────────────────────────
BASE_DIR = Path(__file__).resolve().parents[1]
PLANT_MODEL_PATH = BASE_DIR / "models" / "best_model.pth"
DISEASE_MODELS_DIR = BASE_DIR / "models"

# IMPORTANT
PLANT_THRESHOLD   = 0.55   # if below this => Unknown Plant
DISEASE_THRESHOLD = 0.30   # if below this => uncertain disease

DEVICE        = torch.device("cuda" if torch.cuda.is_available() else "cpu")
IMG_SIZE      = 224
IMAGENET_MEAN = (0.485, 0.456, 0.406)
IMAGENET_STD  = (0.229, 0.224, 0.225)
ALLOWED_TYPES = {"image/jpeg", "image/png", "image/webp", "image/bmp"}
MAX_FILE_SIZE = 10 * 1024 * 1024  # 10 MB

# ⚠️ VERY IMPORTANT:
# THIS ORDER MUST MATCH EXACTLY THE ORDER USED DURING TRAINING
# Replace this with your REAL training class order if needed
PLANT_CLASSES = ["Unknown", "Sugarcane", "Tomato", "Rice", "Potato", "Corn"]

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
# PLANT CLASSIFIER
# ─────────────────────────────────────────────────────────────────────────────
class PlantClassifier(nn.Module):
    def __init__(self, num_classes: int = 6, dropout: float = 0.3):
        super().__init__()
        self.backbone = timm.create_model(
            "efficientnet_b2",
            pretrained=False,
            num_classes=0,
            global_pool="avg"
        )
        in_features = self.backbone.num_features
        self.head = nn.Sequential(
            nn.BatchNorm1d(in_features),
            nn.Dropout(dropout),
            nn.Linear(in_features, 512),
            nn.SiLU(),
            nn.BatchNorm1d(512),
            nn.Dropout(dropout * 0.5),
            nn.Linear(512, num_classes),
        )

    def forward(self, x):
        return self.head(self.backbone(x))


def load_plant_model(path: Path) -> Optional[PlantClassifier]:
    if not Path(path).exists():
        logger.warning(f"Plant model not found: {path}")
        return None

    try:
        model = PlantClassifier(num_classes=len(PLANT_CLASSES))
        ckpt = torch.load(path, map_location=DEVICE)

        # support both plain state_dict and checkpoint dict
        state = ckpt.get("model_state", ckpt)

        model.load_state_dict(state, strict=True)
        model = model.to(DEVICE)
        model.eval()

        torch.set_num_threads(2)
        torch.set_grad_enabled(False)

        logger.info(f"✅ Plant model loaded | device: {DEVICE}")
        logger.info(f"📌 Plant class order: {PLANT_CLASSES}")

        return model

    except Exception as e:
        logger.error(f"❌ Plant model load failed: {e}", exc_info=True)
        return None


plant_model: Optional[PlantClassifier] = load_plant_model(PLANT_MODEL_PATH)

# ─────────────────────────────────────────────────────────────────────────────
# DISEASE MODELS (Keras)
# ─────────────────────────────────────────────────────────────────────────────
disease_models = {}
disease_input_sizes = {}

def load_disease_models():
    try:
        import tensorflow as tf
        tf.config.set_visible_devices([], "GPU")  # force CPU
    except ImportError:
        logger.warning("TensorFlow not installed. Use: pip install tensorflow-cpu")
        return

    for plant in DISEASE_CLASSES:
        model_file = DISEASE_MODELS_DIR / f"{plant.lower()}.keras"

        if not model_file.exists():
            logger.warning(f"Disease model not found: {model_file}")
            continue

        try:
            import tensorflow as tf
            model = tf.keras.models.load_model(str(model_file), compile=False)

            disease_models[plant] = model

            shape = model.input_shape
            if isinstance(shape, list):
                shape = shape[0]

            h = int(shape[1]) if shape[1] else 224
            w = int(shape[2]) if shape[2] else 224
            disease_input_sizes[plant] = (h, w)

            logger.info(f"✅ Disease model loaded: {plant} | input=({h},{w})")

        except Exception as e:
            logger.error(f"❌ Failed to load disease model for {plant}: {e}", exc_info=True)


load_disease_models()

# ─────────────────────────────────────────────────────────────────────────────
# IMAGE TRANSFORMS
# ─────────────────────────────────────────────────────────────────────────────
base_tf = A.Compose([
    A.SmallestMaxSize(256),
    A.CenterCrop(IMG_SIZE, IMG_SIZE),
    A.Normalize(IMAGENET_MEAN, IMAGENET_STD),
    ToTensorV2(),
])

tta_tfs = [
    A.Compose([
        A.SmallestMaxSize(256),
        A.CenterCrop(IMG_SIZE, IMG_SIZE),
        A.Normalize(IMAGENET_MEAN, IMAGENET_STD),
        ToTensorV2(),
    ]),
    A.Compose([
        A.SmallestMaxSize(256),
        A.CenterCrop(IMG_SIZE, IMG_SIZE),
        A.HorizontalFlip(p=1),
        A.Normalize(IMAGENET_MEAN, IMAGENET_STD),
        ToTensorV2(),
    ]),
    A.Compose([
        A.SmallestMaxSize(280),
        A.CenterCrop(IMG_SIZE, IMG_SIZE),
        A.Normalize(IMAGENET_MEAN, IMAGENET_STD),
        ToTensorV2(),
    ]),
]

# ─────────────────────────────────────────────────────────────────────────────
# STEP 1 — PLANT IDENTIFICATION
# ─────────────────────────────────────────────────────────────────────────────
def identify_plant(img_array: np.ndarray, use_tta: bool = False) -> dict:
    if plant_model is None:
        raise RuntimeError("Plant model not loaded")

    t0 = time.perf_counter()

    def _preprocess_fallback(*, smallest_max: int, flip: bool) -> torch.Tensor:
        """
        Fallback preprocessing using PIL + numpy + torch.
        """
        img = Image.fromarray(img_array).convert("RGB")
        w, h = img.size
        short_side = min(w, h)
        scale = (smallest_max / short_side) if short_side else 1.0
        new_w = max(1, int(round(w * scale)))
        new_h = max(1, int(round(h * scale)))
        img = img.resize((new_w, new_h), resample=Image.BILINEAR)

        # Center crop to IMG_SIZE x IMG_SIZE
        left = max(0, (new_w - IMG_SIZE) // 2)
        top = max(0, (new_h - IMG_SIZE) // 2)
        img = img.crop((left, top, left + IMG_SIZE, top + IMG_SIZE))

        if flip:
            img = img.transpose(Image.FLIP_LEFT_RIGHT)

        arr = np.asarray(img).astype(np.float32) / 255.0  # HWC in [0,1]
        arr = arr.transpose(2, 0, 1)  # CHW

        mean = np.asarray(IMAGENET_MEAN, dtype=np.float32)[:, None, None]
        std = np.asarray(IMAGENET_STD, dtype=np.float32)[:, None, None]
        arr = (arr - mean) / std

        return torch.from_numpy(arr).unsqueeze(0)  # [1,3,H,W]

    with torch.no_grad():
        if use_tta:
            probs_list = []
            if _ALBU_AVAILABLE and tta_tfs:
                for tfm in tta_tfs:
                    tensor = tfm(image=img_array)["image"].unsqueeze(0).to(DEVICE)
                    logits = plant_model(tensor)
                    probs = F.softmax(logits, dim=1).squeeze(0).cpu().numpy()
                    probs_list.append(probs)
            else:
                # Same TTA intent as the albumentations pipeline:
                # 1) smallest_max=256 (no flip), 2) smallest_max=256 (flip), 3) smallest_max=280
                for smallest_max, flip in [(256, False), (256, True), (280, False)]:
                    tensor = _preprocess_fallback(smallest_max=smallest_max, flip=flip).to(DEVICE)
                    logits = plant_model(tensor)
                    probs = F.softmax(logits, dim=1).squeeze(0).cpu().numpy()
                    probs_list.append(probs)
            probs = np.mean(probs_list, axis=0)
        else:
            if _ALBU_AVAILABLE and base_tf is not None:
                tensor = base_tf(image=img_array)["image"].unsqueeze(0).to(DEVICE)
                logits = plant_model(tensor)
                probs = F.softmax(logits, dim=1).squeeze(0).cpu().numpy()
            else:
                tensor = _preprocess_fallback(smallest_max=256, flip=False).to(DEVICE)
                logits = plant_model(tensor)
                probs = F.softmax(logits, dim=1).squeeze(0).cpu().numpy()

    elapsed = (time.perf_counter() - t0) * 1000

    idx = int(np.argmax(probs))
    conf = float(probs[idx])
    pred_plant = PLANT_CLASSES[idx]

    # FINAL ACCEPTANCE RULE
    accepted = (pred_plant in SUPPORTED_PLANTS) and (conf >= PLANT_THRESHOLD)

    # If not accepted -> force Unknown
    final_plant = pred_plant if accepted else "Unknown"

    return {
        "predicted_class": pred_plant,
        "plant": final_plant,
        "confidence": conf,
        "all_probs": {cls: round(float(p), 6) for cls, p in zip(PLANT_CLASSES, probs)},
        "elapsed_ms": round(elapsed, 1),
        "accepted": accepted,
    }

# ─────────────────────────────────────────────────────────────────────────────
# STEP 2 — DISEASE DETECTION
# ─────────────────────────────────────────────────────────────────────────────
def detect_disease(img_array: np.ndarray, plant: str) -> dict:
    if plant not in disease_models:
        return {
            "disease": None,
            "confidence": None,
            "accepted": False,
            "reason": f"Disease model not available for {plant}",
        }

    disease_class_list = DISEASE_CLASSES[plant]
    h, w = disease_input_sizes.get(plant, (224, 224))

    img_resized = np.array(
        Image.fromarray(img_array).resize((w, h))
    ).astype(np.float32) / 255.0

    inp = np.expand_dims(img_resized, axis=0)

    t0 = time.perf_counter()
    preds = disease_models[plant].predict(inp, verbose=0)[0]
    elapsed = (time.perf_counter() - t0) * 1000

    idx = int(np.argmax(preds))
    conf = float(preds[idx])

    raw_name = disease_class_list[idx] if idx < len(disease_class_list) else f"Class_{idx}"
    display_name = DISEASE_DISPLAY.get(raw_name, raw_name)

    all_probs = {
        DISEASE_DISPLAY.get(disease_class_list[i], disease_class_list[i]): round(float(p), 6)
        for i, p in enumerate(preds)
        if i < len(disease_class_list)
    }

    return {
        "disease": display_name,
        "raw_class": raw_name,
        "confidence": conf,
        "all_probs": all_probs,
        "elapsed_ms": round(elapsed, 1),
        "accepted": conf >= DISEASE_THRESHOLD,
        "reason": None if conf >= DISEASE_THRESHOLD else f"Low confidence ({conf*100:.1f}%)",
    }

# ─────────────────────────────────────────────────────────────────────────────
# FULL PIPELINE
# ─────────────────────────────────────────────────────────────────────────────
def run_pipeline(img_array: np.ndarray, use_tta: bool = False) -> dict:
    # STEP 1: PLANT IDENTIFICATION
    plant_result = identify_plant(img_array, use_tta=use_tta)

    # If below threshold OR predicted Unknown -> STOP HERE
    if not plant_result["accepted"]:
        return {
            "stage": "plant_identification",
            "plant": {
                "name": "Unknown",
                "predicted_class": plant_result["predicted_class"],
                "confidence": plant_result["confidence"],
                "all_probs": plant_result["all_probs"],
                "emoji": "❓",
                "accepted": False,
            },
            "disease": None,
            "treatment": None,
            "total_ms": plant_result["elapsed_ms"],
            "message": f"Plant not confidently identified (threshold {PLANT_THRESHOLD:.2f}).",
        }

    # Accepted plant (but choose best disease model among top-2 candidates).
    plant_probs = plant_result.get("all_probs") or {}
    plant_top1 = plant_result.get("plant")

    # Consider top-2 supported plants only; this reduces "Tomato misread as Corn".
    candidates = sorted(
        list(SUPPORTED_PLANTS),
        key=lambda p: float(plant_probs.get(p, 0.0)),
        reverse=True,
    )[:2]

    # Avoid extremely low-probability candidates.
    candidate_min = PLANT_THRESHOLD * 0.6
    filtered = [p for p in candidates if float(plant_probs.get(p, 0.0)) >= candidate_min]
    candidates = filtered if filtered else ([plant_top1] if plant_top1 in SUPPORTED_PLANTS else [])

    best_plant = None
    best_disease_result = None
    best_score = -1.0
    total_ms = plant_result["elapsed_ms"]

    for cand in candidates:
        dr = detect_disease(img_array, cand)
        cand_plant_p = float(plant_probs.get(cand, 0.0))
        cand_disease_p = float(dr.get("confidence") or 0.0)
        score = cand_plant_p * cand_disease_p

        if best_disease_result is None:
            best_plant = cand
            best_disease_result = dr
            best_score = score
            continue

        # Prefer accepted disease; if both accepted (or both not), pick higher score.
        if (dr.get("accepted") and not best_disease_result.get("accepted")) or (
            bool(dr.get("accepted")) == bool(best_disease_result.get("accepted")) and score > best_score
        ):
            best_plant = cand
            best_disease_result = dr
            best_score = score

        total_ms += float(dr.get("elapsed_ms") or 0.0)

    plant_name = best_plant or plant_top1
    info = CLASS_INFO.get(plant_name, {})
    disease_result = best_disease_result or detect_disease(img_array, plant_name)

    # total_ms already accumulated above (plant + disease checks)

    # STEP 3: TREATMENT LOOKUP
    treatment = None
    if disease_result.get("accepted") and disease_result.get("raw_class"):
        treatment = get_treatment(plant_name, disease_result["raw_class"])
        if treatment is None:
            logger.warning(f"⚠️ No treatment found for {plant_name} | {disease_result['raw_class']}")

    return {
        "stage": "complete",
        "plant": {
            "name": plant_name,
            "predicted_class": plant_name,
            "confidence": float(plant_probs.get(plant_name, plant_result.get("confidence", 0.0))),
            "all_probs": plant_result["all_probs"],
            "emoji": info.get("emoji", "🌱"),
            "accepted": True,
        },
        "disease": {
            "name": disease_result["disease"],
            "raw_class": disease_result.get("raw_class"),
            "confidence": disease_result["confidence"],
            "all_probs": disease_result.get("all_probs", {}),
            "accepted": disease_result["accepted"],
            "reason": disease_result.get("reason"),
        } if disease_result.get("disease") and disease_result.get("accepted") else None,
        "treatment": treatment,
        "total_ms": round(total_ms, 1),
        "message": None,
    }


def _normalize_severity(raw: Optional[str]) -> str:
    value = (raw or "").strip().lower()
    if "severe" in value or "high" in value:
        return "high"
    if "moderate" in value or "medium" in value:
        return "medium"
    if "none" in value or "healthy" in value or "low" in value:
        return "low"
    return "medium"


def _build_prevention(treatment: Optional[dict]) -> list[str]:
    if not treatment:
        return []
    prevention: list[str] = []
    cultural = treatment.get("cultural")
    if isinstance(cultural, list):
        prevention.extend([str(item) for item in cultural])
    elif cultural:
        prevention.append(str(cultural))
    organic = treatment.get("organic")
    if organic:
        prevention.append(f"Organic: {organic}")
    return prevention


def _build_treatment_list(treatment: Optional[dict]) -> list[str]:
    if not treatment:
        return []
    lines: list[str] = []
    if treatment.get("cause"):
        lines.append(f"Cause: {treatment['cause']}")
    if treatment.get("fertilizer"):
        lines.append(f"Fertilizer: {treatment['fertilizer']}")
    fungicide = treatment.get("fungicide")
    if isinstance(fungicide, list):
        lines.extend([f"Fungicide: {item}" for item in fungicide if item])
    elif fungicide:
        lines.append(f"Fungicide: {fungicide}")
    return lines


def predict_plant_disease(image_bytes: bytes, filename: str = "", tta: bool = False) -> dict:
    """
    Backend service helper used by `main.py`.
    Returns a frontend-compatible shape while preserving the full pipeline payload.
    """
    if plant_model is None:
        raise RuntimeError("Plant model not loaded.")

    if not image_bytes:
        raise RuntimeError("Empty image payload.")

    img = Image.open(io.BytesIO(image_bytes)).convert("RGB")
    img_array = np.array(img)
    pipeline = run_pipeline(img_array, use_tta=tta)

    stage = pipeline.get("stage")
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
        "disease": disease_name,
        "severity": _normalize_severity((treatment_info or {}).get("severity")) if disease_info else "low",
        "treatment": _build_treatment_list(treatment_info),
        "prevention": _build_prevention(treatment_info),
        "plant": pipeline.get("plant", {}),
        "pipeline": pipeline,
        "filename": filename,
    }

    if not result["treatment"] and disease_name == "Unknown":
        if stage == "plant_identification":
            result["treatment"] = ["Plant could not be confidently identified. Upload a clearer leaf image."]
            result["prevention"] = ["Capture image in daylight, focus on affected area, and avoid blur."]
        else:
            result["treatment"] = ["Disease confidence is low. Please re-upload a clearer image of the affected area."]
            result["prevention"] = ["Ensure good lighting, keep the leaf fully visible, and focus on symptoms."]

    return result

# ─────────────────────────────────────────────────────────────────────────────
# FASTAPI APP
# ─────────────────────────────────────────────────────────────────────────────
app = FastAPI(
    title="🌿 PlantSense API v4",
    description="Plant ID → Disease Detection → Treatment Advice",
    version="4.0.0",
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

# ─────────────────────────────────────────────────────────────────────────────
# ROUTES
# ─────────────────────────────────────────────────────────────────────────────
@app.get("/health")
async def health():
    return {
        "status": "ok",
        "plant_model_loaded": plant_model is not None,
        "plant_classes": PLANT_CLASSES,
        "supported_plants": list(SUPPORTED_PLANTS),
        "disease_models_loaded": list(disease_models.keys()),
        "disease_input_sizes": disease_input_sizes,
        "device": str(DEVICE),
        "plant_threshold": PLANT_THRESHOLD,
        "disease_threshold": DISEASE_THRESHOLD,
    }


@app.post("/predict")
async def predict_endpoint(
    file: UploadFile = File(...),
    tta: bool = False,
):
    if file.content_type not in ALLOWED_TYPES:
        raise HTTPException(
            status_code=400,
            detail=f"Unsupported type: {file.content_type}. Use JPEG / PNG / WebP / BMP."
        )

    raw = await file.read()

    if len(raw) > MAX_FILE_SIZE:
        raise HTTPException(status_code=413, detail="File too large. Max 10 MB.")

    if plant_model is None:
        raise HTTPException(status_code=503, detail="Plant model not loaded.")

    try:
        img = Image.open(io.BytesIO(raw)).convert("RGB")
        img_array = np.array(img)

        result = run_pipeline(img_array, use_tta=tta)

        # Thumbnail for UI history
        thumb = img.resize((224, 224))
        buf = io.BytesIO()
        thumb.save(buf, format="JPEG", quality=80)

        result["thumbnail"] = "data:image/jpeg;base64," + base64.b64encode(buf.getvalue()).decode()
        result["original_size"] = {"w": img.width, "h": img.height}
        result["filename"] = file.filename

        return result

    except RuntimeError as e:
        raise HTTPException(status_code=503, detail=str(e))
    except Exception as e:
        logger.error(f"Prediction error: {e}", exc_info=True)
        raise HTTPException(status_code=500, detail=f"Prediction failed: {str(e)}")


@app.get("/classes")
async def get_classes():
    return {
        "plants": PLANT_CLASSES,
        "supported_plants": list(SUPPORTED_PLANTS),
        "disease_classes": DISEASE_CLASSES,
        "disease_display": DISEASE_DISPLAY,
        "plant_threshold": PLANT_THRESHOLD,
        "disease_threshold": DISEASE_THRESHOLD,
    }


@app.post("/reload-models")
async def reload_models():
    global plant_model
    plant_model = load_plant_model(PLANT_MODEL_PATH)
    disease_models.clear()
    disease_input_sizes.clear()
    load_disease_models()

    return {
        "plant_model": plant_model is not None,
        "disease_models": list(disease_models.keys()),
        "disease_input_sizes": disease_input_sizes,
    }