"""API REST en FastAPI y servidor de archivos estáticos para FinGuard AI."""

from pathlib import Path
from typing import Any
from fastapi import FastAPI, Query
from fastapi.middleware.cors import CORSMiddleware
from fastapi.responses import FileResponse
from fastapi.staticfiles import StaticFiles
from pydantic import BaseModel, Field

from backend.ml_engine import FraudDetectionEngine

BASE_DIR = Path(__file__).resolve().parent.parent
FRONTEND_DIR = BASE_DIR / "frontend"

app = FastAPI(
    title="FinGuard AI API",
    description="Sistema inteligente de detección de fraude financiero y anomalías transaccionales (Bloque 2)",
    version="1.0.0",
)

app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

engine = FraudDetectionEngine()


class TransactionInput(BaseModel):
    monto_pen: float = Field(..., gt=0, description="Monto de la transacción en Soles (PEN)")
    hora_dia: int = Field(..., ge=0, le=23, description="Hora del día (0 a 23)")
    distancia_km: float = Field(..., ge=0, description="Distancia desde ubicación habitual (km)")
    intentos_previos_1h: int = Field(..., ge=0, description="Transacciones en la última hora")
    es_internacional: int = Field(..., ge=0, le=1, description="0 = Perú, 1 = Extranjero")
    dispositivo_nuevo: int = Field(..., ge=0, le=1, description="0 = Conocido, 1 = Nuevo")
    ratio_monto_promedio: float = Field(..., gt=0, description="Ratio frente al ticket histórico")
    antiguedad_cuenta_dias: int = Field(..., ge=1, description="Días desde apertura de cuenta")
    threshold: float = Field(0.50, ge=0.05, le=0.95, description="Umbral de decisión")


@app.on_event("startup")
def startup_event() -> None:
    engine.train_all(force_retrain=False)


@app.get("/api/metrics")
def get_metrics(threshold: float = Query(0.50, ge=0.05, le=0.95)) -> dict[str, Any]:
    return engine.evaluate_at_threshold(threshold=threshold)


@app.post("/api/retrain")
def retrain_models(threshold: float = Query(0.50, ge=0.05, le=0.95)) -> dict[str, Any]:
    engine.load_and_prepare_data(force_regenerate=True)
    engine.train_all(force_retrain=True)
    return engine.evaluate_at_threshold(threshold=threshold)


@app.post("/api/predict")
def predict_single_transaction(payload: TransactionInput) -> dict[str, Any]:
    tx_dict = payload.model_dump()
    threshold = tx_dict.pop("threshold", 0.50)
    return engine.predict_transaction(tx_dict, threshold=threshold)


if FRONTEND_DIR.exists():
    app.mount("/static", StaticFiles(directory=str(FRONTEND_DIR)), name="static")

    @app.get("/")
    def serve_index() -> FileResponse:
        return FileResponse(str(FRONTEND_DIR / "index.html"))

