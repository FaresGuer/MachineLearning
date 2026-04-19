from __future__ import annotations

import json
from pathlib import Path
from typing import Any

import joblib
import pandas as pd
from fastapi import FastAPI, HTTPException
from pydantic import BaseModel, Field
from sklearn.neighbors import NearestNeighbors


BASE_DIR = Path(__file__).resolve().parent
ARTIFACTS_DIR = BASE_DIR / "artifacts"


class FeaturesPayload(BaseModel):
    features: dict[str, float]


class RecommendationPayload(BaseModel):
    player_name: str = Field(min_length=1)
    top_n: int = Field(default=5, ge=1, le=50)
    value_tolerance: float = Field(default=0.30, ge=0.0, le=1.0)


class AppState:
    dso1_model: Any = None
    dso1_scaler: Any = None
    dso1_features: list[str] = []

    dso2_model: Any = None
    dso2_scaler: Any = None
    dso2_features: list[str] = []
    dso2_label_encoder: Any = None

    rec_df: pd.DataFrame | None = None
    rec_features: list[str] = []
    nn_model: NearestNeighbors | None = None


state = AppState()
app = FastAPI(
    title="ML Sport API",
    description="API de prediction DSO1/DSO2 et recommandation DSO4.",
    version="1.0.0",
)


def _load_json(path: Path) -> list[str]:
    if not path.exists():
        raise FileNotFoundError(f"Fichier introuvable: {path}")
    with path.open("r", encoding="utf-8") as f:
        value = json.load(f)
    if not isinstance(value, list):
        raise ValueError(f"Format invalide dans {path}. Un tableau JSON est attendu.")
    return [str(item) for item in value]


def _validate_input(features: dict[str, float], expected_features: list[str]) -> list[float]:
    missing = [f for f in expected_features if f not in features]
    extra = [f for f in features if f not in expected_features]

    if missing or extra:
        detail = {
            "missing_features": missing,
            "extra_features": extra,
            "expected_count": len(expected_features),
            "received_count": len(features),
        }
        raise HTTPException(status_code=422, detail=detail)

    return [float(features[name]) for name in expected_features]


def _safe_load(path: Path, required: bool = True) -> Any:
    if not path.exists():
        if required:
            raise FileNotFoundError(f"Artifact manquant: {path}")
        return None
    return joblib.load(path)


def _n_features(obj: Any) -> int | None:
    return getattr(obj, "n_features_in_", None)


def _check_prediction_stack(
    expected_features: list[str],
    scaler: Any,
    model: Any,
    stack_name: str,
) -> None:
    expected_count = len(expected_features)
    scaler_count = _n_features(scaler)
    model_count = _n_features(model)

    if scaler_count is not None and scaler_count != expected_count:
        raise HTTPException(
            status_code=500,
            detail={
                "error": f"Configuration invalide {stack_name}",
                "reason": "features vs scaler incompatibles",
                "features_count": expected_count,
                "scaler_n_features_in": scaler_count,
            },
        )

    if model_count is not None and model_count != expected_count:
        raise HTTPException(
            status_code=500,
            detail={
                "error": f"Configuration invalide {stack_name}",
                "reason": "features vs model incompatibles",
                "features_count": expected_count,
                "model_n_features_in": model_count,
            },
        )


@app.on_event("startup")
def load_artifacts() -> None:
    try:
        state.dso1_model = _safe_load(ARTIFACTS_DIR / "modele_retenu_dso1.pkl")
        state.dso1_scaler = _safe_load(ARTIFACTS_DIR / "scaler_dso1.pkl")
        state.dso1_features = _load_json(ARTIFACTS_DIR / "features_dso1.json")

        state.dso2_model = _safe_load(ARTIFACTS_DIR / "modele_retenu_dso2.pkl")
        state.dso2_scaler = _safe_load(ARTIFACTS_DIR / "scaler_dso2.pkl")
        state.dso2_features = _load_json(ARTIFACTS_DIR / "features_dso2.json")
        state.dso2_label_encoder = _safe_load(ARTIFACTS_DIR / "label_encoder_dso2.pkl", required=False)

        rec_path = ARTIFACTS_DIR / "rec_df.parquet"
        if rec_path.exists():
            state.rec_df = pd.read_parquet(rec_path)
            state.rec_features = _load_json(ARTIFACTS_DIR / "rec_features.json")

            state.nn_model = NearestNeighbors(metric="cosine", algorithm="brute")
            state.nn_model.fit(state.rec_df[state.rec_features].values)
    except Exception as exc:
        raise RuntimeError(
            "Echec de chargement des artifacts. Place les fichiers dans ./artifacts avant de lancer l'API."
        ) from exc


@app.get("/health")
def health() -> dict[str, Any]:
    return {
        "status": "ok",
        "dso1_loaded": state.dso1_model is not None,
        "dso2_loaded": state.dso2_model is not None,
        "dso4_loaded": state.nn_model is not None,
    }


@app.get("/")
def root() -> dict[str, str]:
    return {
        "message": "ML Sport API is running",
        "docs": "/docs",
        "health": "/health",
    }


@app.post("/predict/value")
def predict_value(payload: FeaturesPayload) -> dict[str, Any]:
    if state.dso1_model is None or state.dso1_scaler is None:
        raise HTTPException(status_code=503, detail="Modele DSO1 non charge")

    _check_prediction_stack(state.dso1_features, state.dso1_scaler, state.dso1_model, "DSO1")

    try:
        ordered_values = _validate_input(payload.features, state.dso1_features)
        X_input = pd.DataFrame([ordered_values], columns=state.dso1_features)
        X_scaled = state.dso1_scaler.transform(X_input)
        prediction = float(state.dso1_model.predict(X_scaled)[0])
    except HTTPException:
        raise
    except Exception as exc:
        raise HTTPException(
            status_code=500,
            detail={
                "error": "Echec prediction DSO1",
                "reason": str(exc),
            },
        ) from exc

    return {"predicted_value": prediction}


@app.post("/predict/position")
def predict_position(payload: FeaturesPayload) -> dict[str, Any]:
    if state.dso2_model is None or state.dso2_scaler is None:
        raise HTTPException(status_code=503, detail="Modele DSO2 non charge")

    _check_prediction_stack(state.dso2_features, state.dso2_scaler, state.dso2_model, "DSO2")

    try:
        ordered_values = _validate_input(payload.features, state.dso2_features)
        X_input = pd.DataFrame([ordered_values], columns=state.dso2_features)
        X_scaled = state.dso2_scaler.transform(X_input)
        pred_code = int(state.dso2_model.predict(X_scaled)[0])
    except HTTPException:
        raise
    except Exception as exc:
        raise HTTPException(
            status_code=500,
            detail={
                "error": "Echec prediction DSO2",
                "reason": str(exc),
            },
        ) from exc

    result: dict[str, Any] = {"predicted_position_code": pred_code}
    if state.dso2_label_encoder is not None:
        try:
            label = str(state.dso2_label_encoder.inverse_transform([pred_code])[0])
            result["predicted_position_label"] = label
        except Exception:
            pass

    return result


@app.post("/recommend/similar")
def recommend_similar(payload: RecommendationPayload) -> dict[str, Any]:
    if state.rec_df is None or state.nn_model is None:
        raise HTTPException(status_code=503, detail="Moteur DSO4 non charge")

    rec_df = state.rec_df
    rec_features = state.rec_features

    mask_exact = rec_df["player_name"].astype(str).str.lower() == payload.player_name.lower()
    if mask_exact.any():
        idx_ref = rec_df[mask_exact].index[0]
    else:
        mask_contains = rec_df["player_name"].astype(str).str.lower().str.contains(payload.player_name.lower(), na=False)
        if not mask_contains.any():
            raise HTTPException(status_code=404, detail=f"Joueur introuvable: {payload.player_name}")
        idx_ref = rec_df[mask_contains].index[0]

    ref_row = rec_df.loc[idx_ref]
    ref_value = float(ref_row["value_clean"])

    n_candidates = min(len(rec_df), max(25, payload.top_n * 6))
    distances, indices = state.nn_model.kneighbors([ref_row[rec_features].values], n_neighbors=n_candidates)

    candidates = rec_df.iloc[indices[0]].copy()
    candidates["similarity"] = 1 - distances[0]
    candidates = candidates[candidates.index != idx_ref]

    lower = ref_value * (1 - payload.value_tolerance)
    upper = ref_value * (1 + payload.value_tolerance)
    candidates = candidates[(candidates["value_clean"] >= lower) & (candidates["value_clean"] <= upper)]

    cols = ["player_name", "similarity", "value_clean"] + rec_features[:5]
    out = candidates.sort_values("similarity", ascending=False).head(payload.top_n)[cols]

    return {
        "reference_player": str(ref_row["player_name"]),
        "count": int(len(out)),
        "recommendations": out.to_dict(orient="records"),
    }
