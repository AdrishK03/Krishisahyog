"""
KrishiSahyog FastAPI backend.
Optimized for Render Free Tier using Hugging Face Inference API.
"""
import os
import logging
import io
from pathlib import Path
from contextlib import asynccontextmanager

from dotenv import load_dotenv
load_dotenv()

from fastapi import FastAPI, Depends, HTTPException, UploadFile, File, Header
from fastapi.middleware.cors import CORSMiddleware
from pydantic import BaseModel

# Internal Imports
from database.db import close_mongo_client
from auth.auth_routes import router as auth_router
from auth.jwt_utils import decode_token
from ml.soil_predictor import InputData as SoilFertilizerInput, predict_soil_fertilizer as run_soil_fertilizer_predict
from ml.plant_predictor import predict_plant_disease as predict_plant_disease_logic

logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

@asynccontextmanager
async def lifespan(app: FastAPI):
    """Startup/shutdown lifecycle hooks."""
    logger.info("Starting up KrishiSahyog Backend...")
    yield
    close_mongo_client()
    logger.info("Shutting down backend")

# --- CORS SETUP ---
_cors_default = "http://localhost:5173,http://localhost:5174,https://krishisahyog.vercel.app"
origins = [x.strip() for x in os.getenv("CORS_ORIGINS", _cors_default).split(",") if x.strip()]

app = FastAPI(
    title="KrishiSahyog API",
    description="ML-powered agriculture advisory backend (Cloud-Inference Mode)",
    version="1.1.0",
    lifespan=lifespan,
)

app.add_middleware(
    CORSMiddleware,
    allow_origins=origins,
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

app.include_router(auth_router)

# --- JWT DEPENDENCY ---
def get_current_user_id(authorization: str | None = Header(default=None, alias="Authorization")) -> str:
    if not authorization or not authorization.startswith("Bearer "):
        raise HTTPException(status_code=401, detail="Not authenticated")
    token = authorization.replace("Bearer ", "").strip()
    payload = decode_token(token)
    if not payload or "sub" not in payload:
        raise HTTPException(status_code=401, detail="Invalid token")
    return payload["sub"]

# --- ML ROUTES ---

@app.post("/predict/plant-disease")
async def plant_disease_endpoint(
    file: UploadFile = File(...),
    tta: bool = False,
    _user_id: str = Depends(get_current_user_id),
):
    """Predict plant disease via Hugging Face Inference API."""
    if not file.content_type or not file.content_type.startswith("image/"):
        raise HTTPException(status_code=400, detail="File must be an image")
    
    try:
        contents = await file.read()
        # Offloads the heavy lifting to plant_predictor which calls HF API
        result = predict_plant_disease_logic(contents, file.filename or "", tta=tta)
        return result
    except RuntimeError as e:
        # Handles 503 errors (model loading) from Hugging Face
        raise HTTPException(status_code=503, detail=str(e))
    except Exception as e:
        logger.error(f"Prediction Error: {e}")
        raise HTTPException(status_code=500, detail="Internal prediction error")

@app.post("/predict/soil-fertilizer")
async def soil_fertilizer_endpoint(
    data: SoilFertilizerInput,
    _user_id: str = Depends(get_current_user_id),
):
    """Recommend fertilizer using local Scikit-Learn model (downloaded from HF)."""
    result = run_soil_fertilizer_predict(data)
    if result.get("status") != "success":
        logger.error("Soil fertilizer prediction failed: %s", result.get("message"))
        raise HTTPException(
            status_code=503,
            detail=result.get("message", "Fertilizer prediction failed"),
        )
    return {
        "recommended_fertilizer": result["fertilizer"],
        "explanation": "Suggested by the fertilizer model based on your sensor readings.",
        "model_source": "HuggingFace-Assets"
    }

@app.get("/")
async def root():
    return {"status": "ok", "message": "KrishiSahyog API backend is ready."}

# --- CHATBOT ---

class ChatRequest(BaseModel):
    message: str
    history: list[dict] | None = None

@app.post("/chat")
async def chat_endpoint(
    req: ChatRequest,
    _user_id: str = Depends(get_current_user_id),
):
    from chatbot.chat import chat
    result = chat(req.message, req.history)
    if result.get("error"):
        raise HTTPException(status_code=503, detail=result["error"])
    return {
        "response": result["response"],
        "provider": result.get("provider", "unknown"),
    }

@app.get("/health")
async def health():
    return {"status": "ok", "inference_mode": "remote_api"}
