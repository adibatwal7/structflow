from fastapi import APIRouter, Request
from fastapi.responses import JSONResponse

router = APIRouter(tags=["health"])


@router.get("/healthz")
async def healthz() -> dict[str, str]:
    """Liveness: is the process running?"""
    return {"status": "ok"}


@router.get("/readyz")
async def readyz(request: Request) -> JSONResponse:
    """Readiness: can it take traffic right now?"""
    ready = bool(getattr(request.app.state, "ready", False))
    return JSONResponse(
        {"status": "ready" if ready else "starting"},
        status_code=200 if ready else 503,
    )
