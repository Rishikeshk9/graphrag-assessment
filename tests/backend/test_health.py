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
    assert "/api/v1/health/model-providers" in paths
    assert "/api/v1/ingest/file" in paths
    assert "/api/v1/ingest/{job_id}/cancel" in paths
    assert "/api/v1/knowledge-base" in paths
    assert "/api/v1/knowledge-base/usage" in paths


def test_model_provider_capabilities_do_not_expose_credentials() -> None:
    response = client.get("/api/v1/health/model-providers")

    assert response.status_code == 200
    body = response.json()
    assert body["default_provider"] in {"local", "openrouter"}
    assert isinstance(body["openrouter_configured"], bool)
    assert body["embedding_provider"] == body["default_provider"]
    assert "api_key" not in body
