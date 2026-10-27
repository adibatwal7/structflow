from typing import Annotated

from fastapi import APIRouter, Body, Depends, Request

from structflow.routing.router import Router
from structflow.schemas.errors import ProblemDetails
from structflow.schemas.requests import InferenceRequest
from structflow.schemas.responses import InferenceResponse

router = APIRouter(tags=["inference"])
_PROBLEM = {"model": ProblemDetails, "content": {"application/problem+json": {}}}


def get_router(request: Request) -> Router:
    engine_router: Router = request.app.state.router
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
    return await engine_router.handle(body, request_id=request.state.request_id)
