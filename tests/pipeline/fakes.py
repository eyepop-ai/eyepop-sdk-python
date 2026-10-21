import asyncio
import base64
import json
import struct
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


Script = Callable[[web.WebSocketResponse, "FakePipelineAPI"], Awaitable[None]]


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


def closes(code: int, reason: bytes = b"") -> Script:
    """A script that closes the socket at once."""

    async def script(ws: web.WebSocketResponse, fake: "FakePipelineAPI") -> None:
        await ws.close(code=code, message=reason)

    return script


async def serve(ws: web.WebSocketResponse, fake: "FakePipelineAPI") -> None:
    """A well-behaved socket of contract 019: setPop, setSource (url, combined and chunked upload), getGraph, ping, quit.

    A source plays `fake.frames` predictions, or ends with an error if its id is in `fake.failing`. A live
    source predicts until it is preempted or the socket closes.
    """
    received = Received()
    fake.received.append(received)
    commands: dict[str, dict[str, Any]] = {}
    open_ids: list[str] = []
    live: dict[str, asyncio.Task[None]] = {}
    waiting: list[str] = []  # chunked uploads not yet sent `send`, in order
    current: str | None = None
    buffers: dict[str, bytearray] = {}

    async def end(id: str, reason: str, message: str = "") -> None:
        open_ids.remove(id)
        await ws.send_json({"type": "end", "id": id, "reason": reason, **({"message": message} if message else {})})
        if not open_ids:
            await ws.send_json({"type": "done"})

    async def predict_live(id: str) -> None:
        n = 0
        while not ws.closed:
            await ws.send_json({"type": "prediction", "id": id, "prediction": {**PREDICTIONS[0], "n": n}})
            n += 1
            await asyncio.sleep(0.01)

    async def play(id: str, command: dict[str, Any]) -> None:
        await ws.send_json({"type": "start", "id": id})
        if command.get("live"):
            live[id] = asyncio.create_task(predict_live(id))
        elif id in fake.failing:
            await end(id, "error", "bad media")
        else:
            for n in range(fake.frames):
                await ws.send_json({"type": "prediction", "id": id, "prediction": {**PREDICTIONS[0], "n": n}})
            await end(id, "eos")

    async def next_upload() -> None:
        nonlocal current
        current = waiting.pop(0) if waiting else None
        if current is not None:
            await ws.send_json({"type": "send", "id": current, "credit": fake.window})

    async def set_source(command: dict[str, Any]) -> None:
        id = command["id"]
        if command.get("mode") == "preempt":
            for other in list(open_ids):
                if other in live:
                    live.pop(other).cancel()
                await end(other, "preempted")
        open_ids.append(id)
        commands[id] = command
        await ws.send_json({"type": "result", "id": id})

    while (message := await receive(ws, received)) is not None:
        if isinstance(message, bytes):
            if message[:1] == b"\x01":
                assert current is not None, "a chunk before `send`"
                buffers[current] += message[1:]
                await ws.send_json({"type": "credit", "id": current, "bytes": len(message) - 1})
            else:
                assert message[:1] == b"\x02"
                (length,) = struct.unpack(">I", message[1:5])
                header = json.loads(message[5 : 5 + length])
                fake.media[header["id"]] = message[5 + length :]
                await set_source(header)
                await play(header["id"], header)
            continue
        id = message.get("id")
        match message["type"]:
            case "setPop":
                fake.pops.append(message["pop"])
                if fake.pop_error:
                    await ws.send_json({"type": "error", "id": id, "error": fake.pop_error, "message": "no such ability"})
                else:
                    await ws.send_json({"type": "result", "id": id})
            case "setSource":
                await set_source(message)
                if "upload" in message:
                    buffers[id] = bytearray()
                    waiting.append(id)
                    if current is None:
                        await next_upload()
                else:
                    await play(id, message)
            case "uploaded":
                if id == current:
                    fake.media[id] = bytes(buffers[id])
                    await next_upload()
                    await play(id, commands[id])
            case "getGraph":
                answer = {"contentType": "text/vnd.graphviz", "text": "digraph {}"}
                if message["format"] != "dot":
                    answer = {"contentType": "image/png", "data": base64.b64encode(b"\x89PNG").decode()}
                await ws.send_json({"type": "result", "id": id, **answer})
            case "ping":
                await ws.send_json({"type": "pong", "id": id})
            case "quit":
                await ws.close()
                return
            case other:
                await ws.send_json({"type": "error", "id": id, "error": "unknown_command", "message": other})
    for task in live.values():
        task.cancel()


class _FakeServer:
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
    """Serves queued `(status, body, headers)` answers first, then `200` with `PREDICTIONS`, or the socket to `script`."""

    def __init__(self, script: Script = serve, autoping: bool = True):
        self.script = script
        self.autoping = autoping
        self.frames = 2
        self.window = 128 * 1024
        self.failing: set[str] = set()
        self.pop_error: str | None = None
        self.pops: list[Any] = []
        self.media: dict[str, bytes] = {}
        self.requests: list[Recorded] = []
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
        await self.script(ws, self)
        return ws

    @staticmethod
    def _answer(queue: list[tuple[int, Any, dict[str, str]]]) -> web.Response:
        status, body, headers = queue.pop(0)
        if isinstance(body, str):
            return web.Response(status=status, text=body, headers=headers)
        return web.json_response(body, status=status, headers=headers)


class FakeComputeAPI(_FakeServer):
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
