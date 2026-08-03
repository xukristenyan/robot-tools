# robot-tools architecture

This document records protocol and execution rules that are easy to violate
while changing the implementation. User-facing setup belongs in the
[README](../README.md), runtime installation details in
[runtimes/README.md](../runtimes/README.md), and the procedure for adding a
service in [ADDING_A_SERVICE.md](ADDING_A_SERVICE.md).

## Compatibility boundaries

A client/server connection has four distinct identities:

- `service_id` identifies the logical API. A client must reject a healthy
  endpoint belonging to another service.
- `wire_protocol_version` identifies the shared transport and envelope
  semantics. Change it when an existing client can no longer decode or
  interpret the wire format.
- `api_version` belongs to one service. Change only that service's version
  when existing request, response, or action semantics become incompatible.
- `package_version` identifies a release. A difference may warn, but must not
  replace wire or service-API compatibility checks.

`GET /health` is the compatibility handshake as well as a readiness probe.
Before inference, clients validate readiness, service identity, wire version,
service API version, and required actions. The CLI `status` command applies
the same rules, so swapped ports cannot appear healthy merely because both
endpoints return HTTP 200.

GraspGen and GraspGenX deliberately have different service identities. Similar
capabilities do not make their request semantics or response contracts
interchangeable.

## Contract and route ownership

The request contract is the single source of truth for an action name. Each
request type declares a literal `action`; handler registration derives its
`POST /<action>` route from that value. Do not maintain a second route-name
table in a server or registry.

Pydantic contracts validate decoded requests and handler responses.
MessagePack carries successful action payloads, with NumPy dtype, shape, and
data preserved by `msgpack-numpy`. Health metadata and error envelopes remain
JSON so operators can inspect them without the SDK.

## Request execution semantics

Model handlers are synchronous and run in a worker thread. The HTTP event loop
must remain free to answer `/health` while inference is running.

All actions in one service process share the same admission capacity: one
executing request and one pending request by default. Capacity belongs to the
worker until the handler actually returns. A client timeout or disconnect
stops waiting but does not cancel GPU work or release the permit early.

Capacity is isolated per service process. Separate services may therefore run
concurrently even on the same GPU. There is no cross-service GPU lock,
persistent task queue, cancellation, or preemption; deployment must validate
peak VRAM and latency for its chosen service combination.

Inference clients do not retry automatically. A repeated request may be
expensive or semantically unsafe, especially when the first request is still
running after a timeout.

## Error boundary

Application errors expose a stable `error_code`, a short `message`, and a
`request_id`. Detailed diagnostics stay in server logs. In particular, an
internal-error response must not reveal tracebacks, local paths, environment
variables, model locations, or request bodies.

Fake mode validates repository-owned behavior: contracts, serialization,
routing, compatibility negotiation, capacity, and error handling. It does not
validate upstream imports, GPU execution, model quality, or real artifact
compatibility; those require each runtime's real-mode checks.
