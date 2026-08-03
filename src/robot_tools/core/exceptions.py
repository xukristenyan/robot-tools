"""Public exceptions raised by robot-tools clients."""

from __future__ import annotations


class RobotToolsError(RuntimeError):
    """Base class for robot-tools client failures."""


class ProtocolMismatchError(RobotToolsError):
    """The remote service does not satisfy the client's wire/API contract."""


class InferenceTimeoutError(RobotToolsError):
    """The client stopped waiting for an inference response."""

    def __init__(self, *, service_url: str, action: str):
        self.service_url = service_url
        self.action = action
        super().__init__(f"Inference timed out at {service_url}/{action}")


class RemoteInferenceError(RobotToolsError):
    """A structured error returned by the remote HTTP service."""

    def __init__(
        self,
        *,
        action_url: str,
        status_code: int,
        error_code: str,
        message: str,
        request_id: str | None = None,
    ):
        self.action_url = action_url
        self.status_code = status_code
        self.error_code = error_code
        self.message = message
        self.request_id = request_id
        request_suffix = f", request_id={request_id}" if request_id else ""
        super().__init__(f"{status_code} {error_code} from {action_url}: {message}{request_suffix}")


class ServiceBusyError(RemoteInferenceError):
    """The service has no free execution or pending-request capacity."""


__all__ = [
    "InferenceTimeoutError",
    "ProtocolMismatchError",
    "RemoteInferenceError",
    "RobotToolsError",
    "ServiceBusyError",
]
