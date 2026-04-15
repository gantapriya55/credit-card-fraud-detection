"""
🚀 FastAPI Backend for Fraud Detection
Exposes REST endpoints that any client (Streamlit, mobile app, etc.) can call.
"""

from fastapi import FastAPI, HTTPException
from fastapi.middleware.cors import CORSMiddleware
from pydantic import BaseModel, Field
from typing import List, Optional
import joblib
import numpy as np
from tensorflow import keras
import warnings
warnings.filterwarnings('ignore')

# ============ INIT APP ============
app = FastAPI(
    title="Credit Card Fraud Detection API",
    description="REST API for real-time fraud detection with explainability",
    version="1.0.0",
    docs_url="/docs",
    redoc_url="/redoc"
)

# Allow Streamlit (or any frontend) to call this API
app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

# ============ LOAD MODELS ON STARTUP ============
print("🔄 Loading models...")
rf = joblib.load('models/random_forest.pkl')
xgb = joblib.load('models/xgboost.pkl')
nn = keras.models.load_model('models/neural_network.keras')
explainer = joblib.load('models/shap_explainer.pkl')
feature_names = joblib.load('models/feature_names.pkl')
rf_res = joblib.load('models/rf_results.pkl')
xgb_res = joblib.load('models/xgb_results.pkl')
nn_res = joblib.load('models/nn_results.pkl')
print("✅ All models loaded!")


# ============ REQUEST / RESPONSE SCHEMAS ============
class TransactionRequest(BaseModel):
    """A single transaction with 30 features (V1-V28, Amount_scaled, Time_scaled)."""
    features: List[float] = Field(..., min_items=30, max_items=30,
                                   description="30 feature values in order")
    model: Optional[str] = Field("xgboost", description="xgboost | random_forest | neural_network | ensemble")
    threshold: Optional[float] = Field(0.5, ge=0.0, le=1.0, description="Decision threshold")

    class Config:
        json_schema_extra = {
            "example": {
                "features": [-1.359, -0.072, 2.536, 1.378, -0.338, 0.462, 0.239,
                             0.098, 0.363, 0.090, -0.551, -0.617, -0.991, -0.311,
                             1.468, -0.470, 0.207, 0.025, 0.403, 0.251, -0.018,
                             0.277, -0.110, 0.066, 0.128, -0.189, 0.133, -0.021,
                             0.244, 0.000],
                "model": "xgboost",
                "threshold": 0.5
            }
        }


class FeatureContribution(BaseModel):
    feature: str
    value: float
    shap_value: float
    direction: str


class PredictionResponse(BaseModel):
    is_fraud: bool
    fraud_probability: float
    model_used: str
    threshold: float
    verdict: str
    top_explanations: List[FeatureContribution]


# ============ ENDPOINTS ============

@app.get("/", tags=["Health"])
def root():
    """Health check endpoint."""
    return {
        "status": "✅ online",
        "service": "Fraud Detection API",
        "version": "1.0.0",
        "docs": "/docs",
        "endpoints": ["/predict", "/models", "/health"]
    }


@app.get("/health", tags=["Health"])
def health():
    """Detailed health check."""
    return {
        "status": "healthy",
        "models_loaded": {
            "random_forest": rf is not None,
            "xgboost": xgb is not None,
            "neural_network": nn is not None,
            "shap_explainer": explainer is not None
        },
        "feature_count": len(feature_names)
    }


@app.get("/models", tags=["Models"])
def list_models():
    """Get info about all available models and their metrics."""
    return {
        "available_models": ["random_forest", "xgboost", "neural_network", "ensemble"],
        "default": "xgboost",
        "metrics": {
            "random_forest": {k: rf_res[k] for k in ['accuracy', 'precision', 'recall', 'f1_score', 'roc_auc', 'pr_auc']},
            "xgboost": {k: xgb_res[k] for k in ['accuracy', 'precision', 'recall', 'f1_score', 'roc_auc', 'pr_auc']},
            "neural_network": {k: nn_res[k] for k in ['accuracy', 'precision', 'recall', 'f1_score', 'roc_auc', 'pr_auc']}
        },
        "feature_names": feature_names
    }


@app.post("/predict", response_model=PredictionResponse, tags=["Prediction"])
def predict(req: TransactionRequest):
    """
    Predict whether a transaction is fraud.
    Returns fraud probability + SHAP explanations for the top features.
    """
    try:
        X = np.array(req.features).reshape(1, -1)

        # Run inference
        if req.model == "xgboost":
            prob = float(xgb.predict_proba(X)[0][1])
            used = "XGBoost"
        elif req.model == "random_forest":
            prob = float(rf.predict_proba(X)[0][1])
            used = "Random Forest"
        elif req.model == "neural_network":
            prob = float(nn.predict(X, verbose=0)[0][0])
            used = "Neural Network"
        elif req.model == "ensemble":
            p1 = float(xgb.predict_proba(X)[0][1])
            p2 = float(rf.predict_proba(X)[0][1])
            p3 = float(nn.predict(X, verbose=0)[0][0])
            prob = (p1 + p2 + p3) / 3
            used = f"Ensemble (XGB={p1:.3f}, RF={p2:.3f}, NN={p3:.3f})"
        else:
            raise HTTPException(status_code=400, detail=f"Unknown model: {req.model}")

        is_fraud = prob >= req.threshold

        # Get SHAP explanations
        shap_values = explainer.shap_values(X)[0]
        contributions = sorted(
            zip(feature_names, X[0], shap_values),
            key=lambda x: abs(x[2]),
            reverse=True
        )[:5]

        explanations = [
            FeatureContribution(
                feature=name,
                value=float(val),
                shap_value=float(shap_val),
                direction="🔴 Toward FRAUD" if shap_val > 0 else "🟢 Toward LEGIT"
            )
            for name, val, shap_val in contributions
        ]

        return PredictionResponse(
            is_fraud=is_fraud,
            fraud_probability=prob,
            model_used=used,
            threshold=req.threshold,
            verdict="🚨 FRAUD DETECTED" if is_fraud else "✅ LEGITIMATE",
            top_explanations=explanations
        )

    except Exception as e:
        raise HTTPException(status_code=500, detail=str(e))


@app.post("/predict_batch", tags=["Prediction"])
def predict_batch(transactions: List[TransactionRequest]):
    """Predict fraud for multiple transactions at once."""
    return [predict(tx) for tx in transactions]