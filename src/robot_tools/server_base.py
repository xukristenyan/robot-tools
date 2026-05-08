from __future__ import annotations

import logging
import threading
import time
from typing import Callable, Type

import msgpack
import msgpack_numpy
import zmq
from pydantic import ValidationError

from robot_tools.contracts.base import BaseRequest, BaseResponse, ErrorResponse

msgpack_numpy.patch()

logger = logging.getLogger(__name__)

HandlerFn = Callable[[BaseRequest], BaseResponse]


def handler(request_cls: Type[BaseRequest]):
    """Mark a method as the handler for a given request class.

    The action name is taken from `request_cls.model_fields['action'].default`,
    so contracts must declare `action: Literal["..."] = "..."`.
    """

    def decorator(fn):
        fn._handler_request_cls = request_cls  # noqa: SLF001
        return fn

    return decorator


class BaseServer:
    """ZMQ REP server with msgpack(+numpy) wire, action-based routing,
    pydantic validation, and a single GPU lock around every handler.

    Subclasses register handlers by decorating methods with `@handler(ReqCls)`.
    """

    def __init__(self, host: str = "127.0.0.1", port: int = 5555):
        self.host = host
        self.port = port
        self._gpu_lock = threading.Lock()
        self._handlers: dict[str, tuple[Type[BaseRequest], HandlerFn]] = {}
        self._register_handlers()

    def _register_handlers(self) -> None:
        for name in dir(self):
            attr = getattr(self, name, None)
            req_cls = getattr(attr, "_handler_request_cls", None)
            if req_cls is None:
                continue
            action = req_cls.model_fields["action"].default
            if not action:
                raise RuntimeError(
                    f"{req_cls.__name__} must declare a default `action` (e.g. "
                    f"action: Literal['foo'] = 'foo')"
                )
            self._handlers[action] = (req_cls, attr)
            logger.info("registered handler: %s -> %s", action, name)

    def serve_forever(self) -> None:
        ctx = zmq.Context.instance()
        socket = ctx.socket(zmq.REP)
        addr = f"tcp://{self.host}:{self.port}"
        socket.bind(addr)
        logger.info("listening on %s", addr)

        try:
            while True:
                raw = socket.recv()
                response = self._dispatch(raw)
                socket.send(msgpack.packb(response.model_dump(), use_bin_type=True))
        except KeyboardInterrupt:
            logger.info("shutting down")
        finally:
            socket.close()

    def _dispatch(self, raw: bytes) -> BaseResponse:
        try:
            data = msgpack.unpackb(raw, raw=False)
        except Exception as e:
            return ErrorResponse(error=f"msgpack decode failed: {e}")

        if not isinstance(data, dict):
            return ErrorResponse(error=f"request must be a dict, got {type(data).__name__}")

        action = data.get("action")
        if action == "health":
            return BaseResponse()
        if action not in self._handlers:
            return ErrorResponse(error=f"unknown action: {action!r}")

        req_cls, fn = self._handlers[action]
        try:
            req = req_cls.model_validate(data)
        except ValidationError as e:
            return ErrorResponse(error=f"validation failed: {e}")

        t0 = time.monotonic()
        try:
            with self._gpu_lock:
                resp = fn(req)
        except Exception as e:
            logger.exception("handler %s failed", action)
            return ErrorResponse(error=f"handler failed: {e}")

        resp.timing_ms = (time.monotonic() - t0) * 1000
        return resp
