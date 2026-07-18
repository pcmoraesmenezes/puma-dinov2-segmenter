"""PUMA model serving API — Phases 1 and 4 of the Deploy DinoV2 epic.

Serves the distilled student (not the original DINOv2) + the trained head, the
only combination that meets the latency DoD (<1s) confirmed by the compression
study on 2026-07-17. Swapping models is as simple as swapping
STUDENT_CHECKPOINT_PATH once a more thoroughly trained version is ready.

/metrics exposes real Prometheus metrics (request count by status, latency
histogram) via a middleware that wraps every route — not just /predict — so
request rate, error rate, and latency percentiles are all derivable in Grafana
with plain PromQL (rate(), histogram_quantile()).
"""
import base64
import io
import os
import sys
import time
from contextlib import asynccontextmanager

import torch
from fastapi import FastAPI, File, HTTPException, Request, UploadFile
from fastapi.responses import JSONResponse, Response
from prometheus_client import CONTENT_TYPE_LATEST, Counter, Histogram, generate_latest
from starlette.middleware.base import BaseHTTPMiddleware
from PIL import Image

REPO_ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
sys.path.append(REPO_ROOT)
sys.path.append(os.path.dirname(os.path.abspath(__file__)))

from load_model import load_head, BACKBONE_INPUT_SIZE
from student_model import StudentBackbone, STUDENT_CHECKPOINT_PATH
from preprocessing import make_transform
from utils.logger import get_logger

logger = get_logger(__name__)

STATE = {}

REQUEST_COUNT = Counter(
    "http_requests_total", "Total HTTP requests", ["endpoint", "status"]
)
REQUEST_LATENCY = Histogram(
    "http_request_latency_seconds", "Request latency in seconds", ["endpoint"],
    buckets=(0.01, 0.05, 0.1, 0.25, 0.5, 0.75, 1.0, 2.0, 5.0),
)


class MetricsMiddleware(BaseHTTPMiddleware):
    """Times and counts every request by path + status — one place, all routes."""

    async def dispatch(self, request: Request, call_next):
        start = time.time()
        status_code = 500
        try:
            response = await call_next(request)
            status_code = response.status_code
            return response
        finally:
            elapsed = time.time() - start
            REQUEST_LATENCY.labels(endpoint=request.url.path).observe(elapsed)
            REQUEST_COUNT.labels(endpoint=request.url.path, status=status_code).inc()


def load_serving_bundle():
    device = torch.device("cpu")

    if not STUDENT_CHECKPOINT_PATH.exists():
        raise FileNotFoundError(f"Student checkpoint not found: {STUDENT_CHECKPOINT_PATH}")

    checkpoint = torch.load(STUDENT_CHECKPOINT_PATH, map_location=device)
    student = StudentBackbone().to(device)
    student.load_state_dict(checkpoint["model_state_dict"])
    student.eval()

    head, _ = load_head(device)
    head.eval()

    logger.info(f"Student loaded — epoch {checkpoint['epoch']}, loss {checkpoint['loss']:.4f}")
    return student, head, device


@asynccontextmanager
async def lifespan(app: FastAPI):
    logger.info("Loading model (distilled student + head) at startup...")
    student, head, device = load_serving_bundle()
    STATE["student"] = student
    STATE["head"] = head
    STATE["device"] = device
    STATE["transform"] = make_transform(BACKBONE_INPUT_SIZE)
    logger.info("Model ready to serve.")
    yield
    STATE.clear()


app = FastAPI(title="PUMA DinoV2 Segmenter API", lifespan=lifespan)
app.add_middleware(MetricsMiddleware)


def mask_to_base64_png(mask_tensor: torch.Tensor) -> str:
    mask_np = mask_tensor.squeeze(0).argmax(dim=0).byte().cpu().numpy()
    img = Image.fromarray(mask_np, mode="L")
    buf = io.BytesIO()
    img.save(buf, format="PNG")
    return base64.b64encode(buf.getvalue()).decode("utf-8")


@app.get("/health")
def health():
    return {"status": "ok", "model_loaded": STATE.get("student") is not None}


@app.get("/metrics")
def metrics():
    return Response(generate_latest(), media_type=CONTENT_TYPE_LATEST)


@app.post("/predict")
async def predict(file: UploadFile = File(...)):
    if STATE.get("student") is None:
        raise HTTPException(status_code=503, detail="Model not loaded")

    contents = await file.read()
    try:
        image = Image.open(io.BytesIO(contents)).convert("RGB")
    except Exception:
        raise HTTPException(status_code=400, detail="Invalid image file")

    tensor = STATE["transform"](image).unsqueeze(0)

    start = time.time()
    with torch.no_grad():
        features = STATE["student"](tensor)
        outputs = STATE["head"](features, upscale=True)
    elapsed = time.time() - start

    return JSONResponse({
        "tissue_mask_png_base64": mask_to_base64_png(outputs["tissue"]),
        "nuclei_mask_png_base64": mask_to_base64_png(outputs["nuclei"]),
        "inference_time_seconds": round(elapsed, 4),
        "model": "knowledge_distillation_student_v1",
    })


if __name__ == "__main__":
    import uvicorn
    uvicorn.run("api:app", host="0.0.0.0", port=8080, reload=False)
