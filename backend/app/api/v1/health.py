from datetime import UTC, datetime

from fastapi import APIRouter

from app.config import get_settings
from app.schemas import HealthResponse, ModelProvidersResponse, ServiceStatus

router = APIRouter(prefix="/health", tags=["health"])


@router.get("/live", response_model=HealthResponse, summary="Liveness probe")
async def liveness() -> HealthResponse:
    """Report whether the API process is running."""
    settings = get_settings()
    return HealthResponse(
        status=ServiceStatus.OK,
        service=settings.app_name,
        timestamp=datetime.now(UTC),
    )


@router.get("/ready", response_model=HealthResponse, summary="Readiness probe")
async def readiness() -> HealthResponse:
    """Report basic API readiness; Compose separately gates store startup."""
    settings = get_settings()
    return HealthResponse(
        status=ServiceStatus.OK,
        service=settings.app_name,
        timestamp=datetime.now(UTC),
        dependencies={"api": ServiceStatus.OK},
    )


@router.get("/model-providers", response_model=ModelProvidersResponse, summary="Model provider capabilities")
async def model_providers() -> ModelProvidersResponse:
    """Expose selectable providers without returning API keys or model credentials."""
    settings = get_settings()
    default_provider = "openrouter" if settings.graph_extraction_provider == "openrouter" else "local"
    return ModelProvidersResponse(
        default_provider=default_provider,
        openrouter_configured=bool(settings.openrouter_api_key),
        embedding_provider=default_provider,
    )
