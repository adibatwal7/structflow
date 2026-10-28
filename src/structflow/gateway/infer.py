from typing import Annotated, Any

from fastapi import APIRouter, Body, Depends, Request

from structflow.routing.router import Router
from structflow.schemas.errors import ErrorCode, ProblemDetails
from structflow.schemas.requests import InferenceRequest
from structflow.schemas.responses import InferenceResponse

from .disconnect import unless_disconnected
from .problems import StructFlowError

router = APIRouter(tags=["inference"])
_PROBLEM: dict[str, Any] = {
    "model": ProblemDetails,
    "content": {"application/problem+json": {}},
}


def get_router(request: Request) -> Router:
    engine_router: Router | None = request.app.state.router
    if engine_router is None or not request.app.state.ready:
        raise StructFlowError(
            ErrorCode.NOT_READY,
            "the model is still loading",
            headers={"Retry-After": "5"},
        )
    return engine_router


@router.post(
    "/v1/infer",
    response_model=InferenceResponse,
    responses={
        422: _PROBLEM,
        503: _PROBLEM,
        504: _PROBLEM,
    },  # documents errors in /docs
)
async def infer(
    body: Annotated[InferenceRequest, Body()],
    request: Request,
    engine_router: Annotated[Router, Depends(get_router)],
) -> InferenceResponse:
    if body.request_id is not None:  # a request_id in the body takes precedence
        request.state.request_id = body.request_id
    work = engine_router.handle(body, request_id=request.state.request_id)
    return await unless_disconnected(request, work)
