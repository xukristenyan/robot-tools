"""Client-side connection profile: which host:port each service is reachable at.

`endpoints.yaml` lives with the *client* project; `services.yaml` lives on the
machine that *runs* the servers. Keeping the two separate is what lets a
service move between machines by editing one line here — nothing else changes.

endpoints.yaml:
    endpoints:
      fastfs:   { host: 127.0.0.1, port: 5556 }
      graspgen: { host: gpu5090.local, port: 5557 }

Prefer stable hostnames (mDNS `.local`, Tailscale) over raw IPs so the file
survives DHCP reassignment.
"""

from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path

import yaml


@dataclass(frozen=True)
class Endpoint:
    service_id: str
    host: str
    port: int


def load_endpoints(path: str | Path) -> dict[str, Endpoint]:
    path = Path(path)
    cfg = yaml.safe_load(path.read_text())
    raw = (cfg or {}).get("endpoints")
    if not raw:
        raise ValueError(f"no `endpoints:` entries in {path}")

    out: dict[str, Endpoint] = {}
    for name, entry in raw.items():
        try:
            out[name] = Endpoint(service_id=name, host=str(entry["host"]), port=int(entry["port"]))
        except (TypeError, KeyError) as e:
            raise ValueError(f"endpoint {name!r} in {path} must be a mapping with `host` and `port`") from e
    return out


def get_endpoint(name: str, path: str | Path) -> Endpoint:
    endpoints = load_endpoints(path)
    if name not in endpoints:
        raise KeyError(f"no endpoint {name!r} in {path}. defined: {list(endpoints)}")
    return endpoints[name]
