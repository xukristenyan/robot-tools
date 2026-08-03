from __future__ import annotations

import logging
import threading
import time
import uuid
from collections.abc import Callable

import uvicorn
from fastapi import FastAPI, Request, Response
from fastapi.responses import JSONResponse
from pydantic import ValidationError
from starlette.concurrency import run_in_threadpool

from robot_tools import __version__
from robot_tools.core import wire
from robot_tools.core.wire import BaseRequest, BaseResponse

logger = logging.getLogger(__name__)

HandlerFn = Callable[[BaseRequest], BaseResponse]


def handler(request_cls: type[BaseRequest]):
    """Mark a method as the handler for a request contract.

    The action is taken from ``request_cls.model_fields["action"].default``.
    """

    def decorator(fn):
        fn._handler_request_cls = request_cls
        return fn

    return decorator


class _AdmissionToken:
    """Transfer one admission permit from an action route to its worker.

    The small state machine prevents both early release after a client
    disconnect and permit leaks if worker submission is cancelled.
    """

    def __init__(self, server: BaseServer):
        self._server = server
        self._lock = threading.Lock()
        self._owner = "route"

    def claim_for_worker(self) -> bool:
        with self._lock:
            if self._owner != "route":
                return False
            self._owner = "worker"
            return True

    def release_from_route(self) -> None:
        self._release("route")

    def release_from_worker(self) -> None:
        self._release("worker")

    def _release(self, owner: str) -> None:
        should_release = False
        with self._lock:
            if self._owner == owner:
                self._owner = "released"
                should_release = True
        if should_release:
            self._server._release_admission()


class BaseServer:
    """FastAPI server for synchronous MessagePack inference handlers.

    Inference handlers run in a worker thread. Each server admits one running
    request plus ``max_pending_requests`` waiters, while health remains outside
    inference capacity and stays responsive.
    """

    def __init__(
        self,
        *,
        service_id: str,
        api_version: str,
        host: str = "127.0.0.1",
        port: int = 5555,
        max_pending_requests: int = 1,
        max_request_bytes: int = 64 * 1024 * 1024,
    ):
        if not service_id:
            raise ValueError("service_id must be a stable non-empty string")
        if not api_version:
            raise ValueError("api_version must be a non-empty string")
        if max_pending_requests < 0:
            raise ValueError("max_pending_requests must be >= 0")
        if max_request_bytes <= 0:
            raise ValueError("max_request_bytes must be > 0")

        self.service_id = service_id
        self.api_version = api_version
        self.host = host
        self.port = port
        self.max_pending_requests = max_pending_requests
        self.max_request_bytes = max_request_bytes

        self._ready = threading.Event()
        self._execution_lock = threading.Lock()
        self._admission = threading.BoundedSemaphore(1 + max_pending_requests)
        self._admission_count_lock = threading.Lock()
        self._admitted_requests = 0
        self._handlers: dict[str, tuple[type[BaseRequest], HandlerFn]] = {}
        self._register_handlers()

    @property
    def admitted_request_count(self) -> int:
        """Number of executing plus pending inference requests."""
        with self._admission_count_lock:
            return self._admitted_requests

    def mark_ready(self) -> None:
        """Publish that model initialization and warmup have completed."""
        self._ready.set()

    def mark_not_ready(self) -> None:
        """Stop advertising readiness without stopping the process."""
        self._ready.clear()

    def _register_handlers(self) -> None:
        for name in dir(self):
            attr = getattr(self, name, None)
            req_cls = getattr(attr, "_handler_request_cls", None)
            if req_cls is None:
                continue
            action = req_cls.model_fields["action"].default
            if not action:
                raise RuntimeError(
                    f"{req_cls.__name__} must declare a default `action` (for example action: Literal['foo'] = 'foo')"
                )
            if action in self._handlers:
                raise RuntimeError(f"duplicate handler action: {action!r}")
            self._handlers[action] = (req_cls, attr)
            logger.info("registered handler: POST /%s -> %s", action, name)

    def _build_app(self) -> FastAPI:
        # NumPy contracts have no useful JSON schema. Validation is performed
        # explicitly after MessagePack decoding, so OpenAPI/docs stay disabled.
        app = FastAPI(docs_url=None, redoc_url=None, openapi_url=None)

        @app.get("/health")
        async def health() -> JSONResponse:
            ready = self._ready.is_set()
            body = {
                "status": "ready" if ready else "not_ready",
                "service_id": self.service_id,
                "wire_protocol_version": wire.WIRE_PROTOCOL_VERSION,
                "api_version": self.api_version,
                "package_version": __version__,
                "actions": sorted(self._handlers),
            }
            return JSONResponse(status_code=200 if ready else 503, content=body)

        for action, (req_cls, fn) in self._handlers.items():
            route_handler = self._make_action_route(action, req_cls, fn)
            app.add_api_route(f"/{action}", route_handler, methods=["POST"])
        return app

    def _make_action_route(
        self,
        action: str,
        req_cls: type[BaseRequest],
        fn: HandlerFn,
    ):
        async def route_handler(request: Request) -> Response:
            request_id = uuid.uuid4().hex

            if not self._ready.is_set():
                return self._error_response(
                    503,
                    "service_not_ready",
                    "service is not ready",
                    request_id,
                    action=action,
                )

            media_type = request.headers.get("content-type", "").split(";", 1)[0]
            if media_type.strip().lower() != wire.CONTENT_TYPE:
                return self._error_response(
                    415,
                    "unsupported_media_type",
                    f"Content-Type must be {wire.CONTENT_TYPE}",
                    request_id,
                    action=action,
                )

            content_length = request.headers.get("content-length")
            if content_length is not None:
                try:
                    declared_size = int(content_length)
                except ValueError:
                    declared_size = 0
                if declared_size > self.max_request_bytes:
                    return self._error_response(
                        413,
                        "request_too_large",
                        "request body exceeds the configured limit",
                        request_id,
                        action=action,
                    )

            raw = bytearray()
            async for chunk in request.stream():
                if len(raw) + len(chunk) > self.max_request_bytes:
                    return self._error_response(
                        413,
                        "request_too_large",
                        "request body exceeds the configured limit",
                        request_id,
                        action=action,
                    )
                raw.extend(chunk)

            try:
                data = wire.unpack(bytes(raw))
                if not isinstance(data, dict):
                    raise TypeError("top-level MessagePack value must be a map")
            except Exception:  # noqa: BLE001 - extension hook errors vary
                return self._error_response(
                    400,
                    "invalid_msgpack",
                    "request body is not valid MessagePack",
                    request_id,
                    action=action,
                )

            try:
                req = req_cls.model_validate(data)
            except ValidationError:
                return self._error_response(
                    422,
                    "validation_failed",
                    "request validation failed",
                    request_id,
                    action=action,
                )

            token = self._acquire_admission()
            if token is None:
                return self._error_response(
                    503,
                    "service_busy",
                    "service is busy",
                    request_id,
                    action=action,
                )

            admitted_at = time.monotonic()
            try:
                response = await run_in_threadpool(
                    self._run_job,
                    token,
                    fn,
                    req,
                )
            except Exception:
                logger.exception(
                    "inference failed request_id=%s service_id=%s action=%s",
                    request_id,
                    self.service_id,
                    action,
                )
                return self._error_response(
                    500,
                    "inference_failed",
                    "inference failed",
                    request_id,
                    action=action,
                    log_rejection=False,
                )
            finally:
                # This releases only if ownership never reached the worker.
                # Once claimed, the worker retains capacity until fn finishes.
                token.release_from_route()

            try:
                if not isinstance(response, BaseResponse):
                    raise TypeError("handler must return BaseResponse")
                response.timing_ms = (time.monotonic() - admitted_at) * 1000
                encoded = wire.pack(response.model_dump())
            except Exception:
                logger.exception(
                    "response serialization failed request_id=%s service_id=%s action=%s",
                    request_id,
                    self.service_id,
                    action,
                )
                return self._error_response(
                    500,
                    "response_serialization_failed",
                    "response serialization failed",
                    request_id,
                    action=action,
                    log_rejection=False,
                )

            return Response(
                content=encoded,
                media_type=wire.CONTENT_TYPE,
                headers={"X-Request-ID": request_id},
            )

        route_handler.__name__ = f"post_{action}"
        return route_handler

    def _acquire_admission(self) -> _AdmissionToken | None:
        if not self._admission.acquire(blocking=False):
            return None
        with self._admission_count_lock:
            self._admitted_requests += 1
        return _AdmissionToken(self)

    def _release_admission(self) -> None:
        with self._admission_count_lock:
            self._admitted_requests -= 1
        self._admission.release()

    def _run_job(
        self,
        token: _AdmissionToken,
        fn: HandlerFn,
        req: BaseRequest,
    ) -> BaseResponse:
        if not token.claim_for_worker():
            raise RuntimeError("inference job was withdrawn before worker start")
        try:
            with self._execution_lock:
                return fn(req)
        finally:
            token.release_from_worker()

    def _error_response(
        self,
        status_code: int,
        error_code: str,
        message: str,
        request_id: str,
        *,
        action: str,
        log_rejection: bool = True,
    ) -> JSONResponse:
        if log_rejection:
            level = logging.INFO if error_code == "service_busy" else logging.WARNING
            logger.log(
                level,
                "request rejected request_id=%s service_id=%s action=%s status_code=%d error_code=%s",
                request_id,
                self.service_id,
                action,
                status_code,
                error_code,
            )
        return JSONResponse(
            status_code=status_code,
            content={
                "error_code": error_code,
                "message": message,
                "request_id": request_id,
            },
            headers={"X-Request-ID": request_id},
        )

    def serve_forever(self) -> None:
        app = self._build_app()
        logger.info("listening on http://%s:%d", self.host, self.port)
        uvicorn.run(
            app,
            host=self.host,
            port=self.port,
            log_level="info",
            workers=1,
        )
