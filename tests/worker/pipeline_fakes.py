"""A fake pipeline API and compute API for pipeline mode, ported from eyepop-sdk-python #285 to the socket's commands.

The socket follows eyepop-instance spec 019, `contracts/socket.md`: setPop,
setSource (url, assetUuid, upload, group; combined and chunked), uploaded,
ping and quit, with the server messages and closes a client sees.
"""
import asyncio
import json
import struct
from dataclasses import dataclass, field
from typing import Any, Self

from aiohttp import WSMsgType, web
from aiohttp.test_utils import TestServer


def prediction(n: int) -> dict[str, Any]:
    return {"timestamp": n, "source_width": 640, "source_height": 480, "objects": [{"classLabel": "person"}]}


@dataclass
class Socket:
    """What one socket's client sent: its upgrade's headers, its commands, and each source's media by member."""

    authorization: str | None
    protocols: list[str]
    commands: list[dict[str, Any]] = field(default_factory=list)
    media: dict[str, dict[int | None, bytes]] = field(default_factory=dict)
    combined: list[str] = field(default_factory=list)

    @property
    def sources(self) -> list[dict[str, Any]]:
        return [c for c in self.commands if c.get("type") == "setSource"]

    @property
    def pops(self) -> list[Any]:
        return [c["pop"] for c in self.commands if c.get("type") == "setPop"]


class _Server:
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


class FakePipelineAPI(_Server):
    """One socket per upgrade, each running its own sources in the order they were set.

    A source plays `frames` predictions and ends eos; a url with `bad` in it
    ends with reason error; a live source predicts until the socket closes.
    Scripted behaviour: `upgrade_answers` refuse upgrades first, `refusals`
    answer setSources with an error, `closes` close a socket after a number
    of its setSources were answered, `idle_after_done` closes it `1000 idle`
    once it has no open source, and `overloaded_live` refuses that many live
    setSources `overloaded` and closes `1011 overloaded`, as max-live does.
    """

    def __init__(self):
        self.frames = 2
        self.window = 16
        self.sockets: list[Socket] = []
        self.upgrade_answers: list[tuple[int, dict[str, str]]] = []
        self.refusals: list[str] = []
        self.pop_error: str | None = None
        self.closes: dict[int, tuple[int, str]] = {}
        self.close_on_pop: tuple[int, str] | None = None
        self.idle_after_done = False
        self.overloaded_live = 0
        app = web.Application()
        app.router.add_get("/api/v1/pipeline", self._upgrade)
        super().__init__(app)

    async def _upgrade(self, request: web.Request) -> web.StreamResponse:
        if self.upgrade_answers:
            status, headers = self.upgrade_answers.pop(0)
            return web.Response(status=status, headers=headers, text="refused")
        socket = Socket(
            request.headers.get("Authorization"),
            [p.strip() for p in request.headers.get("Sec-WebSocket-Protocol", "").split(",") if p.strip()],
        )
        self.sockets.append(socket)
        ws = web.WebSocketResponse(protocols=("eyepop.v1",))
        await ws.prepare(request)
        await _Run(self, socket, ws).serve()
        return ws


class _Run:
    """One socket's server side."""

    def __init__(self, fake: FakePipelineAPI, socket: Socket, ws: web.WebSocketResponse):
        self.fake = fake
        self.socket = socket
        self.ws = ws
        self.open: list[str] = []
        self.answered = 0
        self.queue: asyncio.Queue[tuple[str, dict[str, Any]] | None] = asyncio.Queue()
        # Chunked uploads in line, as (id, member); the first one is sending.
        self.uploads: list[tuple[str, int | None]] = []
        self.missing: dict[str, int] = {}
        self.commands: dict[str, dict[str, Any]] = {}

    async def serve(self):
        player = asyncio.create_task(self.play())
        try:
            while not self.ws.closed:
                message = await self.ws.receive()
                if message.type == WSMsgType.TEXT:
                    await self.command(json.loads(message.data))
                elif message.type == WSMsgType.BINARY:
                    await self.binary(message.data)
                else:
                    break
        finally:
            player.cancel()
            await asyncio.gather(player, return_exceptions=True)

    async def send(self, message: dict[str, Any]):
        if not self.ws.closed:
            await self.ws.send_json(message)

    async def close(self, code: int, reason: str):
        await self.ws.close(code=code, message=reason.encode())

    async def command(self, command: dict[str, Any]):
        self.socket.commands.append(command)
        id_: Any = command.get("id")
        match command.get("type"):
            case "setPop":
                if self.fake.close_on_pop is not None:
                    code, reason = self.fake.close_on_pop
                    self.fake.close_on_pop = None
                    await self.close(code, reason)
                elif self.fake.pop_error:
                    await self.send({"type": "error", "id": id_, "error": self.fake.pop_error, "message": "no such ability"})
                else:
                    await self.send({"type": "result", "id": id_})
            case "setSource":
                await self.set_source(command, None)
            case "uploaded":
                key = (id_, command.get("member"))
                if self.uploads and self.uploads[0] == key:
                    self.uploads.pop(0)
                    self.missing[id_] -= 1
                    if self.missing[id_] == 0:
                        await self.queue.put((id_, self.commands[id_]))
                    await self.next_upload()
            case "ping":
                await self.send({"type": "pong", "id": id_})
            case "quit":
                await self.ws.close()
            case other:
                await self.send({"type": "error", "id": id_, "error": "unknown_command", "message": other})

    async def binary(self, data: bytes):
        if data[:1] == b"\x01":
            id_, member = self.uploads[0]
            media = self.socket.media[id_]
            media[member] = media.get(member, b"") + data[1:]
            await self.send({"type": "credit", "id": id_, "bytes": len(data) - 1, **_member(member)})
            return
        assert data[:1] == b"\x02", "a binary message starts with a kind byte"
        (length,) = struct.unpack(">I", data[1:5])
        command = json.loads(data[5 : 5 + length])
        self.socket.commands.append(command)
        self.socket.combined.append(command["id"])
        await self.set_source(command, data[5 + length :])

    async def set_source(self, command: dict[str, Any], combined: bytes | None):
        id_ = command["id"]
        if self.fake.refusals:
            await self.send({"type": "error", "id": id_, "error": self.fake.refusals.pop(0), "message": "refused"})
            return
        live = command.get("live") or str(command.get("url", "")).startswith(("rtsp://", "rtmp://"))
        if live and self.fake.overloaded_live > 0:
            self.fake.overloaded_live -= 1
            await self.send({"type": "error", "id": id_, "error": "overloaded", "message": "max-live"})
            await self.close(1011, "overloaded")
            return
        self.open.append(id_)
        self.commands[id_] = command
        self.socket.media[id_] = {}
        await self.send({"type": "result", "id": id_})
        self.answered += 1
        if (close := self.fake.closes.pop(self.answered, None)) is not None:
            await self.close(*close)
            return
        members = _upload_members(command)
        if combined is not None:
            for member, size in members:
                self.socket.media[id_][member] = combined[:size] if size is not None else combined
                combined = combined[size:] if size is not None else b""
            await self.queue.put((id_, command))
        elif members:
            self.missing[id_] = len(members)
            sending = bool(self.uploads)
            self.uploads.extend((id_, member) for member, _ in members)
            if not sending:
                await self.next_upload()
        else:
            await self.queue.put((id_, command))

    async def next_upload(self):
        if self.uploads:
            id_, member = self.uploads[0]
            await self.send({"type": "send", "id": id_, "credit": self.fake.window, **_member(member)})

    async def play(self):
        while (item := await self.queue.get()) is not None:
            id_, command = item
            await self.send({"type": "start", "id": id_})
            if command.get("live") or str(command.get("url", "")).startswith(("rtsp://", "rtmp://")):
                n = 0
                while not self.ws.closed:
                    await self.send({"type": "prediction", "id": id_, "prediction": prediction(n)})
                    n += 1
                    await asyncio.sleep(0.01)
                return
            if "bad" in str(command.get("url", "")):
                await self.end(id_, "error", "bad media")
                continue
            for n in range(self.fake.frames):
                await self.send({"type": "prediction", "id": id_, "prediction": prediction(n)})
            await self.end(id_, "eos")

    async def end(self, id_: str, reason: str, message: str = ""):
        self.open.remove(id_)
        await self.send({"type": "end", "id": id_, "reason": reason, **({"message": message} if message else {})})
        if not self.open:
            await self.send({"type": "done"})
            if self.fake.idle_after_done:
                await self.close(1000, "idle")


def _member(member: int | None) -> dict[str, int]:
    return {} if member is None else {"member": member}


def _upload_members(command: dict[str, Any]) -> list[tuple[int | None, int | None]]:
    """Each upload member's index (None for a single upload) and its combined bytes."""
    if "upload" in command:
        return [(None, None)]
    return [
        (i, member["upload"].get("bytes"))
        for i, member in enumerate(command.get("group", []))
        if "upload" in member
    ]


class FakeComputeAPI(_Server):
    """Serves `/v1/configs` with `pipeline_api_url`, and mints a new token at each API key exchange."""

    def __init__(self, pipeline_api_url: str | None, expires_in: int = 3600):
        self.pipeline_api_url = pipeline_api_url
        self.expires_in = expires_in
        self.exchanges: list[str | None] = []
        self.config_authorizations: list[str | None] = []
        app = web.Application()
        app.router.add_post("/v1/auth/authenticate", self._exchange)
        app.router.add_get("/v1/configs", self._configs)
        super().__init__(app)

    @property
    def minted(self) -> list[str]:
        return [f"minted-{n}" for n in range(1, len(self.exchanges) + 1)]

    async def _exchange(self, request: web.Request) -> web.Response:
        self.exchanges.append(request.headers.get("Authorization"))
        return web.json_response({"access_token": f"minted-{len(self.exchanges)}", "expires_in": self.expires_in})

    async def _configs(self, request: web.Request) -> web.Response:
        self.config_authorizations.append(request.headers.get("Authorization"))
        config: dict[str, Any] = {"dataset_api_url": "https://dataset-api.example"}
        if self.pipeline_api_url is not None:
            config["pipeline_api_url"] = self.pipeline_api_url
        return web.json_response(config)
