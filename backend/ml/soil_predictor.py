"""
Soil fertilizer recommendation module.
Loads model from backend/models/soil_fertilizer_model.pkl or uses rule-based dummy logic.
"""
from pathlib import Path

MODEL_PATH = Path(__file__).resolve().parent.parent / "models" / "soil_fertilizer_model.pkl"

_model = None


def _load_model():
    global _model
    if _model is not None:
        return _model
    if not MODEL_PATH.exists():
        return None
    try:
        with open(MODEL_PATH, "rb") as f:
            import pickle
            _model = pickle.load(f)
        return _model
    except Exception:
        return None


def _dummy_recommend(n: float, p: float, k: float, ph: float, moisture: float, temp: float) -> tuple[str, str]:
    """Rule-based fertilizer recommendation when model is unavailable."""
    recommendations = []
    if n < 50:
        recommendations.append("Urea (N) - nitrogen deficient")
    if p < 30:
        recommendations.append("DAP or SSP - phosphorus deficient")
    if k < 150:
        recommendations.append("MOP (Potash) - potassium deficient")
    if ph < 6.0:
        recommendations.append("Lime - soil too acidic")
    elif ph > 7.5:
        recommendations.append("Gypsum - soil too alkaline")
    if moisture < 30:
        recommendations.append("Increase irrigation")
    if temp > 35:
        recommendations.append("Avoid fertilization during extreme heat")

    if not recommendations:
        rec = "NPK 20-20-20 (balanced) - soil parameters optimal"
        expl = "Your soil parameters are within optimal ranges. A balanced fertilizer is recommended for maintenance."
    else:
        rec = "; ".join(recommendations)
        expl = f"Based on your soil data: N={n}, P={p}, K={k}, pH={ph}, moisture={moisture}%, temp={temp}°C. " + rec
    return rec, expl


def predict_soil_fertilizer(
    nitrogen: float,
    phosphorus: float,
    potassium: float,
    ph: float,
    moisture: float,
    temperature: float,
) -> dict:
    """
    Recommend fertilizer based on soil parameters.
    Returns: {recommended_fertilizer, explanation, model_used}
    """
    model = _load_model()
    if model is not None:
        try:
            import numpy as np
            X = np.array([[nitrogen, phosphorus, potassium, ph, moisture, temperature]])
            if hasattr(model, "predict"):
                pred = model.predict(X)[0]
                rec = str(pred)
                expl = f"Model recommendation based on N={nitrogen}, P={phosphorus}, K={potassium}, pH={ph}, moisture={moisture}%, temp={temperature}°C."
                return {
                    "recommended_fertilizer": rec,
                    "explanation": expl,
                    "model_used": "real",
                }
        except Exception:
            pass

    rec, expl = _dummy_recommend(nitrogen, phosphorus, potassium, ph, moisture, temperature)
    return {
        "recommended_fertilizer": rec,
        "explanation": expl,
        "model_used": "dummy",
    }
