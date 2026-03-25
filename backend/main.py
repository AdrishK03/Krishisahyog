"""
KrishiSahyog FastAPI backend.
Run: uvicorn main:app --host 127.0.0.1 --port 8000 --reload
"""
from dotenv import load_dotenv
# Load .env before any other imports that use env vars
load_dotenv()
import os
import logging
from contextlib import asynccontextmanager



logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)


@asynccontextmanager
async def lifespan(app: object):
    """Startup: create DB tables. Shutdown: cleanup."""
    try:
        from database.db import engine, Base
        from database import models  # noqa: F401 - register models with Base
        Base.metadata.create_all(bind=engine)
        logger.info("Database tables created/verified")
    except Exception as e:
        logger.error("Database initialization failed: %s", e)
        raise
    yield
    logger.info("Shutting down backend")


# Build app after env is loaded
from fastapi import FastAPI, Depends, HTTPException, UploadFile, File, Header
from fastapi.middleware.cors import CORSMiddleware
from pydantic import BaseModel

from database.db import get_db
from auth.auth_routes import router as auth_router
from auth.jwt_utils import decode_token

# CORS origins - include both localhost and 127.0.0.1 for dev
_cors_default = "http://localhost:5173,http://127.0.0.1:5173,http://localhost:5174,http://127.0.0.1:5174"
origins = [x.strip() for x in os.getenv("CORS_ORIGINS", _cors_default).split(",") if x.strip()]

app = FastAPI(
    title="KrishiSahyog API",
    description="ML-powered agriculture advisory backend",
    version="1.0.0",
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


# --- JWT dependency ---
def get_current_user_id(authorization: str | None = None) -> str:
    """Extract and validate JWT from Authorization header."""
    if not authorization or not authorization.startswith("Bearer "):
        raise HTTPException(status_code=401, detail="Not authenticated")
    token = authorization.replace("Bearer ", "").strip()
    payload = decode_token(token)
    if not payload or "sub" not in payload:
        raise HTTPException(status_code=401, detail="Invalid token")
    return payload["sub"]


def require_auth(authorization: str | None = Header(default=None, alias="Authorization")):
    return get_current_user_id(authorization)


# --- ML Routes ---

@app.post("/predict/plant-disease")
def predict_plant_disease(
    file: UploadFile = File(...),
    _user_id: str = Depends(require_auth),
):
    """Predict plant disease from uploaded image. Requires auth."""
    if not file.content_type or not file.content_type.startswith("image/"):
        raise HTTPException(status_code=400, detail="File must be an image")
    contents = file.file.read()
    from ml.plant_predictor import predict_plant_disease as predict
    result = predict(contents, file.filename or "")
    # Enrich with treatment/prevention for frontend compatibility
    if result.get("model_used") == "dummy":
        result["disease"] = result["prediction"]
        result["severity"] = "medium"
        result["treatment"] = [
            "Upload a trained model to backend/models/plant_disease_model.pkl for real predictions.",
        ]
        result["prevention"] = ["Ensure model file exists and is valid."]
    else:
        result["disease"] = result["prediction"]
        result["severity"] = "medium"
        result["treatment"] = ["Follow recommended agricultural practices for the detected condition."]
        result["prevention"] = ["Practice crop rotation and maintain plant health."]
    return result


class SoilInput(BaseModel):
    nitrogen: float
    phosphorus: float
    potassium: float
    ph: float
    moisture: float
    temperature: float


@app.post("/predict/soil-fertilizer")
def predict_soil_fertilizer(
    data: SoilInput,
    _user_id: str = Depends(require_auth),
):
    """Recommend fertilizer based on soil parameters. Requires auth."""
    from ml.soil_predictor import predict_soil_fertilizer as predict
    return predict(
        nitrogen=data.nitrogen,
        phosphorus=data.phosphorus,
        potassium=data.potassium,
        ph=data.ph,
        moisture=data.moisture,
        temperature=data.temperature,
    )


# --- Chatbot ---

class ChatRequest(BaseModel):
    message: str
    history: list[dict] | None = None


@app.post("/chat")
def chat_endpoint(
    req: ChatRequest,
    _user_id: str = Depends(require_auth),
):
    """Agriculture chatbot with Claude → Gemini → OpenAI fallback. Requires auth."""
    from chatbot.chat import chat
    result = chat(req.message, req.history)
    if result.get("error"):
        # 503 = all providers failed, 429 = quota hit but at least one worked partially
        status = 503
        raise HTTPException(status_code=status, detail=result["error"])
    return {
        "response": result["response"],
        "provider": result.get("provider", "unknown"),
    }


# --- Health ---

@app.get("/health")
def health():
    """Health check — also shows which AI providers are configured."""
    configured = []
    if os.getenv("ANTHROPIC_API_KEY"):
        configured.append("claude")
    if os.getenv("GEMINI_API_KEY"):
        configured.append("gemini")
    if os.getenv("OPENAI_API_KEY"):
        configured.append("openai")
    return {"status": "ok", "ai_providers": configured}