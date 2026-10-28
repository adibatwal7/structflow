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
    state = request.app.state
    if getattr(state, "ready", False):
        return JSONResponse({"status": "ready"})
    error = getattr(state, "startup_error", None)
    if error:
        return JSONResponse({"status": "failed", "detail": error}, status_code=503)
    return JSONResponse({"status": "starting"}, status_code=503)
