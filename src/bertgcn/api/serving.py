#!/usr/bin/env python3
"""
Production Model Serving API for BertGCN

FastAPI-based REST API for serving BertGCN models with:
- Async request handling
- Request/response validation
- Health checks and monitoring
- Rate limiting
- Model versioning
- Batch prediction support
"""

import asyncio
import time
from contextlib import asynccontextmanager
from typing import Any, Dict, List, Optional

import mlflow
import torch
from fastapi import BackgroundTasks, Depends, FastAPI, HTTPException
from fastapi.middleware.cors import CORSMiddleware
from fastapi.middleware.gzip import GZipMiddleware
from prometheus_client import CONTENT_TYPE_LATEST, Counter, Histogram, generate_latest
from prometheus_fastapi_instrumentator import Instrumentator
from pydantic import BaseModel, Field

from bertgcn.core.inference import ModelPredictor
from bertgcn.core.models import BertGCN
from bertgcn.utils.monitoring import log_prediction_metrics
from bertgcn.utils.rate_limiter import RateLimiter

# Prometheus metrics
PREDICTION_COUNTER = Counter("bertgcn_predictions_total", "Total predictions made")
PREDICTION_LATENCY = Histogram(
    "bertgcn_prediction_duration_seconds", "Prediction latency"
)
ERROR_COUNTER = Counter("bertgcn_errors_total", "Total errors", ["error_type"])

# Global model store
model_store = {}


@asynccontextmanager
async def lifespan(app: FastAPI):
    """Application lifespan manager."""
    # Startup
    await load_models()
    yield
    # Shutdown
    await cleanup_models()


app = FastAPI(
    title="BertGCN Clinical Text Classification API",
    description="Production API for clinical text classification using BertGCN",
    version="1.0.0",
    lifespan=lifespan,
)

# Add middleware
app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)
app.add_middleware(GZipMiddleware, minimum_size=1000)

# Setup Prometheus monitoring
instrumentator = Instrumentator()
instrumentator.instrument(app).expose(app)


# Request/Response models
class TextInput(BaseModel):
    text: str = Field(
        ..., min_length=1, max_length=10000, description="Input clinical text"
    )
    document_level: str = Field("letter", description="Document level for processing")


class BatchTextInput(BaseModel):
    texts: List[str] = Field(
        ..., min_items=1, max_items=100, description="Batch of clinical texts"
    )
    document_level: str = Field("letter", description="Document level for processing")


class PredictionResponse(BaseModel):
    prediction: str = Field(..., description="Predicted class label")
    confidence: float = Field(
        ..., ge=0, le=1, description="Prediction confidence score"
    )
    probabilities: Dict[str, float] = Field(..., description="Class probabilities")
    processing_time: float = Field(..., description="Processing time in seconds")
    model_version: str = Field(..., description="Model version used")


class BatchPredictionResponse(BaseModel):
    predictions: List[PredictionResponse] = Field(..., description="Batch predictions")
    total_processing_time: float = Field(..., description="Total processing time")
    batch_size: int = Field(..., description="Number of items processed")


class HealthResponse(BaseModel):
    status: str = Field(..., description="Service health status")
    timestamp: float = Field(..., description="Health check timestamp")
    model_loaded: bool = Field(..., description="Whether model is loaded")
    gpu_available: bool = Field(..., description="Whether GPU is available")
    memory_usage: Dict[str, float] = Field(..., description="Memory usage statistics")


# Dependency injection
async def get_rate_limiter() -> RateLimiter:
    """Get rate limiter instance."""
    return RateLimiter(max_requests=100, window_seconds=60)


async def get_model_predictor(model_version: str = "latest") -> ModelPredictor:
    """Get model predictor instance."""
    if model_version not in model_store:
        raise HTTPException(
            status_code=404, detail=f"Model version {model_version} not found"
        )
    return model_store[model_version]


# API Routes
@app.get("/health", response_model=HealthResponse)
async def health_check():
    """Health check endpoint."""
    import psutil

    return HealthResponse(
        status="healthy",
        timestamp=time.time(),
        model_loaded=len(model_store) > 0,
        gpu_available=torch.cuda.is_available(),
        memory_usage={
            "cpu_percent": psutil.cpu_percent(),
            "memory_percent": psutil.virtual_memory().percent,
            "gpu_memory_used": (
                torch.cuda.memory_allocated() / 1024**3
                if torch.cuda.is_available()
                else 0
            ),
        },
    )


@app.get("/ready")
async def readiness_check():
    """Readiness check for Kubernetes."""
    if len(model_store) == 0:
        raise HTTPException(status_code=503, detail="Models not loaded")
    return {"status": "ready"}


@app.get("/models")
async def list_models():
    """List available model versions."""
    return {"available_models": list(model_store.keys()), "default_model": "latest"}


@app.post("/predict", response_model=PredictionResponse)
async def predict_text(
    input_data: TextInput,
    background_tasks: BackgroundTasks,
    rate_limiter: RateLimiter = Depends(get_rate_limiter),
    predictor: ModelPredictor = Depends(get_model_predictor),
):
    """Make a single prediction."""
    # Rate limiting
    await rate_limiter.check_rate_limit()

    start_time = time.time()

    try:
        with PREDICTION_LATENCY.time():
            # Make prediction
            result = await predictor.predict_async(
                text=input_data.text, document_level=input_data.document_level
            )

            processing_time = time.time() - start_time

            response = PredictionResponse(
                prediction=result["prediction"],
                confidence=result["confidence"],
                probabilities=result["probabilities"],
                processing_time=processing_time,
                model_version=predictor.model_version,
            )

            # Background logging
            background_tasks.add_task(
                log_prediction_metrics, input_data.text, result, processing_time
            )

            PREDICTION_COUNTER.inc()
            return response

    except Exception as e:
        ERROR_COUNTER.labels(error_type="prediction_error").inc()
        raise HTTPException(status_code=500, detail=f"Prediction failed: {str(e)}")


@app.post("/predict/batch", response_model=BatchPredictionResponse)
async def predict_batch(
    input_data: BatchTextInput,
    background_tasks: BackgroundTasks,
    rate_limiter: RateLimiter = Depends(get_rate_limiter),
    predictor: ModelPredictor = Depends(get_model_predictor),
):
    """Make batch predictions."""
    # Rate limiting (check per item in batch)
    for _ in input_data.texts:
        await rate_limiter.check_rate_limit()

    start_time = time.time()

    try:
        # Make batch predictions
        results = await predictor.predict_batch_async(
            texts=input_data.texts, document_level=input_data.document_level
        )

        total_time = time.time() - start_time

        predictions = [
            PredictionResponse(
                prediction=result["prediction"],
                confidence=result["confidence"],
                probabilities=result["probabilities"],
                processing_time=result["processing_time"],
                model_version=predictor.model_version,
            )
            for result in results
        ]

        response = BatchPredictionResponse(
            predictions=predictions,
            total_processing_time=total_time,
            batch_size=len(input_data.texts),
        )

        # Background logging
        background_tasks.add_task(
            log_prediction_metrics, input_data.texts, results, total_time
        )

        PREDICTION_COUNTER.inc(len(input_data.texts))
        return response

    except Exception as e:
        ERROR_COUNTER.labels(error_type="batch_prediction_error").inc()
        raise HTTPException(
            status_code=500, detail=f"Batch prediction failed: {str(e)}"
        )


@app.get("/metrics")
async def get_metrics():
    """Prometheus metrics endpoint."""
    return Response(generate_latest(), media_type=CONTENT_TYPE_LATEST)


# Model management functions
async def load_models():
    """Load all available models."""
    try:
        # Load latest model from MLflow
        model_uri = "models:/bertgcn_clinical/latest"
        model = mlflow.pytorch.load_model(model_uri)

        predictor = ModelPredictor(model, model_version="latest")
        model_store["latest"] = predictor

        print(f"✅ Loaded model: latest")

    except Exception as e:
        print(f"❌ Failed to load model: {str(e)}")


async def cleanup_models():
    """Cleanup models on shutdown."""
    for version, predictor in model_store.items():
        await predictor.cleanup()
    model_store.clear()


if __name__ == "__main__":
    import uvicorn

    uvicorn.run(
        "serving:app",
        host="0.0.0.0",
        port=8000,
        workers=4,
        reload=False,
        access_log=True,
    )
