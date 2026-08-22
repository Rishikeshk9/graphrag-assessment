from datetime import UTC, datetime

from fastapi import APIRouter

from app.config import get_settings
from app.schemas import HealthResponse, ServiceStatus

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
    """Report API readiness; external dependency probes arrive in later stages."""
    settings = get_settings()
    return HealthResponse(
        status=ServiceStatus.OK,
        service=settings.app_name,
        timestamp=datetime.now(UTC),
        dependencies={"api": ServiceStatus.OK},
    )

