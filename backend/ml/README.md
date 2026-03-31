# 🌿 PlantSense — Plant Classification API

EfficientNetB2-powered plant classifier with a FastAPI backend and beautiful web UI.

## Files

```
plant_app/
├── main.py              ← FastAPI backend
├── requirements.txt     ← Python dependencies
├── static/
│   └── index.html       ← Frontend UI
└── Run_PlantSense_Colab.ipynb  ← One-click Colab launcher
```

## Option A — Run in Google Colab (easiest)

1. Upload `Run_PlantSense_Colab.ipynb` to Colab
2. Set your `MODEL_PATH` (Drive path or upload)
3. Get a free ngrok token at https://ngrok.com → paste in the last cell
4. Run all cells → get a public URL → open in browser

## Option B — Run Locally

```bash
# 1. Install dependencies
pip install -r requirements.txt

# 2. Run the server
MODEL_PATH=/path/to/best_model.pth uvicorn main:app --reload --host 0.0.0.0 --port 8000

# 3. Open browser
http://localhost:8000
```

## API Endpoints

| Method | Endpoint | Description |
|--------|----------|-------------|
| GET    | `/`         | Frontend UI |
| GET    | `/health`   | Model status, device |
| POST   | `/predict`  | Upload image → classification |
| GET    | `/docs`     | Interactive Swagger API docs |
| GET    | `/classes`  | List all classes + metadata |

### POST /predict

**Parameters:**
- `file` (form-data): image file — JPEG, PNG, WebP, BMP (max 10 MB)
- `tta` (query, bool): enable test-time augmentation (default: `true`)

**Response:**
```json
{
  "class": "Tomato",
  "class_idx": 2,
  "confidence": 0.9234,
  "all_probs": {
    "Unknown": 0.002,
    "Sugarcane": 0.001,
    "Tomato": 0.923,
    "Rice": 0.040,
    "Potato": 0.028,
    "Corn": 0.006
  },
  "inference_ms": 42.3,
  "used_tta": true,
  "device": "cuda",
  "info": {
    "emoji": "🍅",
    "color": "#ef4444",
    "desc": "Solanum lycopersicum — fruit/vegetable crop"
  },
  "thumbnail": "data:image/jpeg;base64,...",
  "original_size": {"w": 1024, "h": 768},
  "filename": "tomato_leaf.jpg"
}
```

## Classes

| # | Class     | Emoji |
|---|-----------|-------|
| 0 | Unknown   | ❓    |
| 1 | Sugarcane | 🌾    |
| 2 | Tomato    | 🍅    |
| 3 | Rice      | 🌾    |
| 4 | Potato    | 🥔    |
| 5 | Corn      | 🌽    |
