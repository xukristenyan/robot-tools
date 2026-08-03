from __future__ import annotations

import logging
import threading
import time
from collections.abc import Mapping
from typing import ClassVar, TypeVar

import httpx
from typing_extensions import Self

from robot_tools import __version__
from robot_tools.core import wire
from robot_tools.core.endpoints import Endpoint
from robot_tools.core.exceptions import (
    InferenceTimeoutError,
    ProtocolMismatchError,
    RemoteInferenceError,
    ServiceBusyError,
)
from robot_tools.core.wire import BaseRequest, BaseResponse

logger = logging.getLogger(__name__)

R = TypeVar("R", bound=BaseResponse)

HEALTH_TIMEOUT_S = 2.0


class BaseClient:
    """Synchronous HTTP client with strict service compatibility checks."""

    SUPPORTED_SERVICE_APIS: ClassVar[Mapping[str, frozenset[str]]] = {}
    REQUIRED_ACTIONS: ClassVar[frozenset[str]] = frozenset()
    SUPPORTED_WIRE_PROTOCOLS: ClassVar[frozenset[str]] = frozenset({wire.WIRE_PROTOCOL_VERSION})

    def __init__(
        self,
        host: str = "localhost",
        port: int = 5555,
        timeout_ms: int = 120_000,
        wait: bool = True,
    ):
        self._base_url = f"http://{host}:{port}"
        self._client = httpx.Client(
            base_url=self._base_url,
            timeout=timeout_ms / 1000,
        )
        self._compatibility_lock = threading.Lock()
        self._compatible = False
        self._server_info: dict | None = None
        if wait:
            self._wait_for_server()

    @classmethod
    def from_endpoint(cls, endpoint: Endpoint, **kwargs) -> Self:
        return cls(host=endpoint.host, port=endpoint.port, **kwargs)

    def _request(self, req: BaseRequest, response_cls: type[R]) -> R:
        self._ensure_compatible()
        action_url = f"{self._base_url}/{req.action}"
        try:
            response = self._client.post(
                f"/{req.action}",
                content=wire.pack(req.model_dump()),
                headers={"Content-Type": wire.CONTENT_TYPE},
            )
        except httpx.TimeoutException as exc:
            raise InferenceTimeoutError(
                service_url=self._base_url,
                action=req.action,
            ) from exc
        except httpx.RequestError as exc:
            raise ConnectionError(f"Request to {action_url} failed: {exc}") from exc

        if response.status_code != 200:
            raise self._remote_error(response, action_url)

        media_type = response.headers.get("content-type", "").split(";", 1)[0]
        if media_type.strip().lower() != wire.CONTENT_TYPE:
            raise ProtocolMismatchError(
                f"{action_url} returned Content-Type "
                f"{response.headers.get('content-type')!r}; expected "
                f"{wire.CONTENT_TYPE!r}"
            )

        try:
            payload = wire.unpack(response.content)
            return response_cls.model_validate(payload)
        except Exception as exc:
            request_id = response.headers.get("X-Request-ID")
            suffix = f" (request_id={request_id})" if request_id else ""
            raise ProtocolMismatchError(f"{action_url} returned an invalid {response_cls.__name__}{suffix}") from exc

    def health(self, timeout_s: float = HEALTH_TIMEOUT_S) -> dict:
        """Return the raw JSON readiness document from ``GET /health``."""
        response = self._client.get("/health", timeout=timeout_s)
        response.raise_for_status()
        return self._decode_health(response)

    def health_check(self) -> bool:
        try:
            info = self.health()
            return info.get("status") == "ready"
        except (httpx.HTTPError, ProtocolMismatchError, ValueError):
            return False

    def check_compatibility(self, timeout_s: float = HEALTH_TIMEOUT_S) -> dict:
        """Validate the remote service and return its health metadata.

        Successful validation is cached and shared with the inference path, so
        callers such as ``robot-tools status`` exercise the same service, wire,
        API, and action checks as a real request without sending inference work.
        """
        if self._compatible:
            assert self._server_info is not None
            return dict(self._server_info)
        with self._compatibility_lock:
            if self._compatible:
                assert self._server_info is not None
                return dict(self._server_info)
            info = self.health(timeout_s=timeout_s)
            self._validate_health(info)
            self._server_info = info
            self._compatible = True
            return dict(info)

    def _ensure_compatible(self) -> None:
        self.check_compatibility()

    def _wait_for_server(
        self,
        interval: float = 2.0,
        max_attempts: int = 60,
    ) -> None:
        logger.info("Connecting to %s ...", self._base_url)
        last_transport_error: Exception | None = None
        for _ in range(max_attempts):
            try:
                response = self._client.get(
                    "/health",
                    timeout=HEALTH_TIMEOUT_S,
                )
            except httpx.RequestError as exc:
                last_transport_error = exc
                time.sleep(interval)
                continue

            if response.status_code == 503:
                time.sleep(interval)
                continue
            if response.status_code != 200:
                raise ProtocolMismatchError(f"{self._base_url}/health returned HTTP {response.status_code}")

            info = self._decode_health(response)
            self._validate_health(info)
            self._server_info = info
            self._compatible = True
            logger.info(
                "Connected to %s (%s API %s)",
                self._base_url,
                info["service_id"],
                info["api_version"],
            )
            return

        raise ConnectionError(f"Server at {self._base_url} did not become ready") from last_transport_error

    def _validate_health(self, info: dict) -> None:
        required = {
            "status",
            "service_id",
            "wire_protocol_version",
            "api_version",
            "package_version",
            "actions",
        }
        missing = required.difference(info)
        if missing:
            raise ProtocolMismatchError(f"{self._base_url}/health is missing fields: {sorted(missing)}")

        for field in (
            "status",
            "service_id",
            "wire_protocol_version",
            "api_version",
            "package_version",
        ):
            if not isinstance(info[field], str):
                raise ProtocolMismatchError(f"{self._base_url}/health has an invalid {field} field; expected a string")
        actions = info["actions"]
        if not isinstance(actions, list) or not all(isinstance(action, str) for action in actions):
            raise ProtocolMismatchError(
                f"{self._base_url}/health has an invalid actions field; expected a list of strings"
            )

        if info["status"] != "ready":
            raise ProtocolMismatchError(f"{self._base_url} is not ready: {info['status']!r}")

        service_id = info["service_id"]
        if service_id not in self.SUPPORTED_SERVICE_APIS:
            allowed = sorted(self.SUPPORTED_SERVICE_APIS)
            raise ProtocolMismatchError(
                f"{type(self).__name__} cannot connect to service {service_id!r}; expected one of {allowed}"
            )

        wire_version = info["wire_protocol_version"]
        if wire_version not in self.SUPPORTED_WIRE_PROTOCOLS:
            raise ProtocolMismatchError(
                f"Unsupported wire protocol {wire_version!r}; supported: {sorted(self.SUPPORTED_WIRE_PROTOCOLS)}"
            )

        api_version = info["api_version"]
        supported_apis = self.SUPPORTED_SERVICE_APIS[service_id]
        if api_version not in supported_apis:
            raise ProtocolMismatchError(
                f"Unsupported {service_id} API {api_version!r}; supported: {sorted(supported_apis)}"
            )

        missing_actions = self.REQUIRED_ACTIONS.difference(actions)
        if missing_actions:
            raise ProtocolMismatchError(f"{service_id} is missing required actions: {sorted(missing_actions)}")

        package_version = info["package_version"]
        if package_version != __version__:
            logger.warning(
                "%s runs robot-tools %s but this client is %s; wire/API checks passed",
                self._base_url,
                package_version,
                __version__,
            )

    @staticmethod
    def _decode_health(response: httpx.Response) -> dict:
        try:
            info = response.json()
        except ValueError as exc:
            raise ProtocolMismatchError(f"{response.request.url} did not return JSON health metadata") from exc
        if not isinstance(info, dict):
            raise ProtocolMismatchError(f"{response.request.url} returned non-object health metadata")
        return info

    @staticmethod
    def _remote_error(
        response: httpx.Response,
        action_url: str,
    ) -> RemoteInferenceError:
        try:
            payload = response.json()
        except ValueError:
            payload = {}
        if not isinstance(payload, dict):
            payload = {}

        error_code = payload.get("error_code", "remote_http_error")
        message = payload.get("message", "remote request failed")
        request_id = payload.get("request_id") or response.headers.get("X-Request-ID")
        error_cls = (
            ServiceBusyError if response.status_code == 503 and error_code == "service_busy" else RemoteInferenceError
        )
        return error_cls(
            action_url=action_url,
            status_code=response.status_code,
            error_code=str(error_code),
            message=str(message),
            request_id=str(request_id) if request_id else None,
        )

    def close(self) -> None:
        self._client.close()

    def __enter__(self) -> Self:
        return self

    def __exit__(self, *args) -> None:
        self.close()
