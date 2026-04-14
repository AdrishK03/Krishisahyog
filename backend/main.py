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
    """Startup/shutdown lifecycle hooks."""
    yield
    from database.db import close_mongo_client
    close_mongo_client()
    logger.info("Shutting down backend")


# Build app after env is loaded
from fastapi import FastAPI, Depends, HTTPException, UploadFile, File, Header
from fastapi.middleware.cors import CORSMiddleware
from pydantic import BaseModel

from database.db import get_db
from auth.auth_routes import router as auth_router
from auth.jwt_utils import decode_token

# CORS origins - include both localhost and 127.0.0.1 for dev
_cors_default = "http://localhost:5173,http://127.0.0.1:5173,http://localhost:5174,http://127.0.0.1:5174,https://krishisahyog.vercel.app"
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

from ml.soil_predictor import InputData as SoilFertilizerInput, predict_soil_fertilizer as run_soil_fertilizer_predict


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
    tta: bool = False,
    _user_id: str = Depends(require_auth),
):
    """Predict plant disease from uploaded image. Requires auth."""
    if not file.content_type or not file.content_type.startswith("image/"):
        raise HTTPException(status_code=400, detail="File must be an image")
    contents = file.file.read()
    from ml.plant_predictor import predict_plant_disease as predict
    return predict(contents, file.filename or "", tta=tta)


@app.post("/predict/soil-fertilizer")
def predict_soil_fertilizer_endpoint(
    data: SoilFertilizerInput,
    _user_id: str = Depends(require_auth),
):
    """Recommend fertilizer from IoT readings (NPK, temp, humidity, moisture) + soil/crop type. Requires auth."""
    result = run_soil_fertilizer_predict(data)
    if result.get("status") != "success":
        raise HTTPException(
            status_code=503,
            detail=result.get("message", "Fertilizer prediction failed"),
        )
    return {
        "recommended_fertilizer": result["fertilizer"],
        "explanation": (
            "Suggested by the fertilizer model using your live sensor readings "
            "and selected soil and crop types."
        ),
        "model_used": "real",
    }


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
async def health():
    return {"status": "ok"}
