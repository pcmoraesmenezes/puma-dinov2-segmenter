"""API integration tests — pytest + TestClient, per the Deploy DinoV2 epic DoD."""
import base64
import io
import os
import sys

import pytest
from fastapi.testclient import TestClient
from PIL import Image

sys.path.append(os.path.dirname(os.path.abspath(__file__)))
from api import app


@pytest.fixture(scope="module")
def client():
    with TestClient(app) as c:
        yield c


def _dummy_image_bytes(size=256):
    img = Image.new("RGB", (size, size), color=(120, 80, 100))
    buf = io.BytesIO()
    img.save(buf, format="PNG")
    buf.seek(0)
    return buf


def test_health_reports_model_loaded(client):
    resp = client.get("/health")
    assert resp.status_code == 200
    body = resp.json()
    assert body["status"] == "ok"
    assert body["model_loaded"] is True


def test_predict_returns_masks_and_timing(client):
    files = {"file": ("test.png", _dummy_image_bytes(), "image/png")}
    resp = client.post("/predict", files=files)
    assert resp.status_code == 200

    body = resp.json()
    assert "tissue_mask_png_base64" in body
    assert "nuclei_mask_png_base64" in body
    assert body["inference_time_seconds"] > 0
    assert body["model"] == "knowledge_distillation_student_v1"

    # confirms the base64 decodes into a valid PNG
    tissue_png = base64.b64decode(body["tissue_mask_png_base64"])
    decoded = Image.open(io.BytesIO(tissue_png))
    assert decoded.size == (1024, 1024)


def test_predict_rejects_invalid_file(client):
    files = {"file": ("not_an_image.txt", io.BytesIO(b"this is not an image"), "text/plain")}
    resp = client.post("/predict", files=files)
    assert resp.status_code == 400


def test_metrics_counts_requests(client):
    resp_before = client.get("/metrics")
    assert resp_before.status_code == 200
    assert "predict_requests_total" in resp_before.text
