from fastapi.testclient import TestClient

from app.main import app


client = TestClient(app)


def test_liveness_contract() -> None:
    response = client.get("/api/v1/health/live")

    assert response.status_code == 200
    body = response.json()
    assert body["status"] == "ok"
    assert body["service"] == "GraphRAG Assessment API"
    assert "timestamp" in body


def test_readiness_contract() -> None:
    response = client.get("/api/v1/health/ready")

    assert response.status_code == 200
    assert response.json()["dependencies"] == {"api": "ok"}


def test_openapi_exposes_health_routes() -> None:
    paths = client.get("/openapi.json").json()["paths"]

    assert "/api/v1/health/live" in paths
    assert "/api/v1/health/ready" in paths
    assert "/api/v1/ingest/file" in paths
