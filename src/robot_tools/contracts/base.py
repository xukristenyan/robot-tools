from __future__ import annotations

from pydantic import BaseModel, ConfigDict


class BaseRequest(BaseModel):
    """Common parent for all request schemas.

    Subclasses must override `action` with a Literal["..."] default so the server
    can route requests by action name without manual type checks.
    """

    model_config = ConfigDict(arbitrary_types_allowed=True)

    action: str


class BaseResponse(BaseModel):
    """Common parent for all response schemas. Carries server-side timing."""

    model_config = ConfigDict(arbitrary_types_allowed=True)

    timing_ms: float | None = None


class ErrorResponse(BaseResponse):
    """Returned by the server when a handler raises or validation fails."""

    error: str
