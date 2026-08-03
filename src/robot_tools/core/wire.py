"""Wire-level schemas and MessagePack encoding shared by clients and servers.

Both client and server import these two functions, so request/response bodies
are always encoded the same way. Arrays keep their dtype, shape, and raw bytes
(no JSON text round-trip, no base64) — that is why bodies are msgpack instead
of FastAPI's default JSON.
"""

from __future__ import annotations

import msgpack
import msgpack_numpy
from pydantic import BaseModel, ConfigDict

CONTENT_TYPE = "application/msgpack"
WIRE_PROTOCOL_VERSION = "1"


class BaseRequest(BaseModel):
    """Base schema for action requests sent over the wire."""

    model_config = ConfigDict(arbitrary_types_allowed=True)

    action: str


class BaseResponse(BaseModel):
    """Base schema for responses returned over the wire."""

    model_config = ConfigDict(arbitrary_types_allowed=True)

    timing_ms: float | None = None


def pack(payload: object) -> bytes:
    return msgpack.packb(payload, default=msgpack_numpy.encode, use_bin_type=True)


def unpack(raw: bytes) -> object:
    return msgpack.unpackb(raw, object_hook=msgpack_numpy.decode, raw=False)


__all__ = [
    "CONTENT_TYPE",
    "WIRE_PROTOCOL_VERSION",
    "BaseRequest",
    "BaseResponse",
    "pack",
    "unpack",
]
