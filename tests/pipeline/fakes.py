"""In-process fakes of the pipeline API and of compute-api's exchange, for the pipeline client's tests.

Each records every request and upgrade it receives. A socket's behaviour is a script: an async
function given the server's socket and the client's first message.
"""

from collections.abc import Awaitable, Callable
from dataclasses import dataclass, field
from typing import Any, Self

from aiohttp import BodyPartReader, WSMsgType, web
from aiohttp.test_utils import TestServer

PREDICTIONS = [{"timestamp": 0, "source_width": 640, "source_height": 480, "objects": [{"classLabel": "person"}]}]


@dataclass
class Recorded:
    method: str
    authorization: str | None
    origin: str | None
    content_type: str
    protocols: list[str] = field(default_factory=list)
    json: Any = None
    parts: dict[str, tuple[str, bytes]] = field(default_factory=dict)


@dataclass
class Received:
    """What a script read from the client: text messages parsed, binary frames as bytes."""

    messages: list[Any] = field(default_factory=list)
    close_code: int | None = None


Script = Callable[[web.WebSocketResponse, dict[str, Any], "FakePipelineAPI"], Awaitable[None]]


async def receive(ws: web.WebSocketResponse, received: Received, timeout: float | None = None) -> Any:
    """The client's next message, recorded; None once the client has closed (or on `timeout`)."""
    try:
        msg = await ws.receive(timeout)
    except TimeoutError:
        return None
    if msg.type == WSMsgType.TEXT:
        received.messages.append(msg.json())
    elif msg.type == WSMsgType.BINARY:
        received.messages.append(msg.data)
    else:
        received.close_code = ws.close_code
        return None
    return received.messages[-1]


async def until_closed(ws: web.WebSocketResponse, received: Received) -> None:
    while await receive(ws, received) is not None:
        pass


async def batch_done(ws: web.WebSocketResponse, request: dict[str, Any], fake: "FakePipelineAPI") -> None:
    """Each item: `start`, one prediction, `end`; then `done` and `1000`."""
    for item in request["items"]:
        await ws.send_json({"type": "start", "id": item["id"]})
        await ws.send_json({"type": "prediction", "id": item["id"], "prediction": PREDICTIONS[0]})
        await ws.send_json({"type": "end", "id": item["id"], "reason": "eos"})
    await ws.send_json({"type": "done"})
    await ws.close()


class _FakeServer:
    """An aiohttp app on a free loopback port, started by `async with`."""

    def __init__(self, app: web.Application):
        self.server = TestServer(app, host="127.0.0.1")

    @property
    def url(self) -> str:
        return str(self.server.make_url("")).rstrip("/")

    async def __aenter__(self) -> Self:
        """Start serving."""
        await self.server.start_server()
        return self

    async def __aexit__(self, *_: object) -> None:
        """Stop serving."""
        await self.server.close()


class FakePipelineAPI(_FakeServer):
    """`POST` and the upgrade at `/api/v1/pipeline`.

    `post_answers` and `upgrade_answers` are queues of `(status, body, headers)` served before the
    default: `200` with `PREDICTIONS`, and an upgrade that runs `script`.
    """

    def __init__(self, script: Script = batch_done, autoping: bool = True):
        self.script = script
        self.autoping = autoping
        self.requests: list[Recorded] = []
        self.socket_requests: list[dict[str, Any]] = []
        self.received: list[Received] = []
        self.post_answers: list[tuple[int, Any, dict[str, str]]] = []
        self.upgrade_answers: list[tuple[int, Any, dict[str, str]]] = []
        app = web.Application()
        app.router.add_route("*", "/api/v1/pipeline", self._handle)
        super().__init__(app)

    @property
    def upgrades(self) -> list[Recorded]:
        return [r for r in self.requests if r.method == "GET"]

    async def _handle(self, request: web.Request) -> web.StreamResponse:
        recorded = Recorded(
            request.method,
            request.headers.get("Authorization"),
            request.headers.get("Origin"),
            request.content_type,
            [p.strip() for p in request.headers.get("Sec-WebSocket-Protocol", "").split(",") if p.strip()],
        )
        self.requests.append(recorded)
        if request.method == "POST":
            if request.content_type == "application/json":
                recorded.json = await request.json()
            else:
                reader = await request.multipart()
                async for part in reader:
                    assert isinstance(part, BodyPartReader) and part.name
                    recorded.parts[part.name] = (part.headers.get("Content-Type", ""), await part.read())
            return self._answer(self.post_answers) if self.post_answers else web.json_response({"predictions": PREDICTIONS})
        if self.upgrade_answers:
            return self._answer(self.upgrade_answers)
        ws = web.WebSocketResponse(protocols=("eyepop.v1",), autoping=self.autoping)
        await ws.prepare(request)
        first = await ws.receive_json()
        self.socket_requests.append(first)
        await self.script(ws, first, self)
        return ws

    @staticmethod
    def _answer(queue: list[tuple[int, Any, dict[str, str]]]) -> web.Response:
        status, body, headers = queue.pop(0)
        if isinstance(body, str):
            return web.Response(status=status, text=body, headers=headers)
        return web.json_response(body, status=status, headers=headers)


class FakeComputeAPI(_FakeServer):
    """compute-api's `POST /v1/auth/authenticate`: mints `minted-1`, `minted-2`, ... valid for `expires_in`."""

    def __init__(self, expires_in: int | None = 3600, status: int = 200):
        self.expires_in = expires_in
        self.status = status
        self.authorizations: list[str | None] = []
        app = web.Application()
        app.router.add_post("/v1/auth/authenticate", self._handle)
        super().__init__(app)

    @property
    def minted(self) -> list[str]:
        return [f"minted-{n}" for n in range(1, len(self.authorizations) + 1)]

    async def _handle(self, request: web.Request) -> web.Response:
        self.authorizations.append(request.headers.get("Authorization"))
        if self.status != 200:
            return web.Response(status=self.status, text="Unauthorized")
        return web.json_response(
            {"access_token": f"minted-{len(self.authorizations)}", "expires_in": self.expires_in}
        )
