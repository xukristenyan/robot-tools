from __future__ import annotations

import logging
import time
from typing import Type, TypeVar

from typing_extensions import Self

import msgpack
import msgpack_numpy
import zmq

from robot_tools.contracts.base import BaseRequest, BaseResponse

msgpack_numpy.patch()

logger = logging.getLogger(__name__)

R = TypeVar("R", bound=BaseResponse)


class BaseClient:
    """ZMQ REQ client with msgpack(+numpy) wire format.

    Subclasses expose typed methods that build a `BaseRequest`, call `_request`
    with the matching response type, and return the validated response.
    """

    def __init__(
        self,
        host: str = "localhost",
        port: int = 5555,
        timeout_ms: int = 120_000,
        wait: bool = True,
    ):
        self._addr = f"tcp://{host}:{port}"
        self._timeout_ms = timeout_ms
        self._ctx = zmq.Context.instance()
        self._socket: zmq.Socket | None = None
        if wait:
            self._wait_for_server()

    def _request(self, req: BaseRequest, response_cls: Type[R]) -> R:
        if self._socket is None:
            self._socket = self._make_socket()

        payload = req.model_dump()
        self._socket.send(msgpack.packb(payload, use_bin_type=True))
        raw = self._socket.recv()
        data = msgpack.unpackb(raw, raw=False)

        if isinstance(data, dict) and "error" in data and "action" not in data:
            raise RuntimeError(f"Server error: {data['error']}")
        return response_cls.model_validate(data)

    def health_check(self) -> bool:
        try:
            req = BaseRequest(action="health")
            self._request(req, BaseResponse)
            return True
        except Exception:
            return False

    def _make_socket(self) -> zmq.Socket:
        s = self._ctx.socket(zmq.REQ)
        s.setsockopt(zmq.RCVTIMEO, self._timeout_ms)
        s.setsockopt(zmq.SNDTIMEO, self._timeout_ms)
        s.setsockopt(zmq.LINGER, 0)
        s.connect(self._addr)
        return s

    def _wait_for_server(self, interval: float = 2.0, max_attempts: int = 60) -> None:
        logger.info("Connecting to %s ...", self._addr)
        for _ in range(max_attempts):
            self._socket = self._make_socket()
            if self.health_check():
                logger.info("Connected to %s", self._addr)
                return
            self._reset_socket()
            time.sleep(interval)
        raise RuntimeError(f"Server at {self._addr} did not become ready")

    def _reset_socket(self) -> None:
        if self._socket is not None:
            self._socket.close()
            self._socket = None

    def close(self) -> None:
        self._reset_socket()

    def __enter__(self) -> Self:
        return self

    def __exit__(self, *args) -> None:
        self.close()
