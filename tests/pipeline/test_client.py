import ast
import asyncio
import logging
import pathlib
import socket
import time
from collections.abc import AsyncIterator
from typing import Any

import pytest
from aiohttp import web

import eyepop.pipeline
from eyepop.pipeline import Ability, EyePop, EyePopError, Model, Pop, Prediction, Result
from tests.pipeline.fakes import PREDICTIONS, FakeComputeAPI, FakePipelineAPI, closes

JWT = "header.payload.signature"
API_KEY = "test-api-key-1234"
ENV_KEY = "env-api-key-never-read"
PERSON = Ability("eyepop.person")
PREDICTION = Prediction.parse(PREDICTIONS[0])


@pytest.fixture(autouse=True)
def env_api_key(monkeypatch: pytest.MonkeyPatch):
    monkeypatch.setenv("EYEPOP_API_KEY", ENV_KEY)


@pytest.fixture
async def pipeline_api() -> AsyncIterator[FakePipelineAPI]:
    """The fake pipeline API; afterwards, every request carried a bearer JWT, no API key and no Origin, and every upgrade offered `eyepop.v1`."""
    async with FakePipelineAPI() as fake:
        yield fake
    for recorded in fake.requests:
        assert recorded.authorization is not None and recorded.authorization.startswith("Bearer ")
        assert ENV_KEY not in recorded.authorization and API_KEY not in recorded.authorization
        assert recorded.origin is None
    for upgrade in fake.upgrades:
        assert "eyepop.v1" in upgrade.protocols


@pytest.fixture
async def compute_api() -> AsyncIterator[FakeComputeAPI]:
    async with FakeComputeAPI() as fake:
        yield fake
    assert all(a == f"Bearer {API_KEY}" for a in fake.authorizations)


def free_port() -> int:
    with socket.socket() as s:
        s.bind(("127.0.0.1", 0))
        return s.getsockname()[1]


async def collect(client: EyePop) -> list[Result]:
    """One image over a socket of its own."""
    return [r async for r in client.use(PERSON).batch(["https://example.com/1.jpg"])]


# One image


async def test_image_by_url_asset_and_file(pipeline_api: FakePipelineAPI, tmp_path: pathlib.Path):
    """A url and an asset go as JSON, a file and bytes as multipart typed from the name or content_type; a str is a url or a path."""
    image = tmp_path / "people.jpg"
    image.write_bytes(b"\xff\xd8 jpeg bytes")
    async with EyePop(pipeline_api.url, access_token=JWT) as client:
        person = client.use(PERSON)
        assert await person.image({"url": "https://example.com/1.jpg"}) == PREDICTION
        assert await client.use(Pop({"components": []})).image({"assetUuid": "4d1e"}) == PREDICTION
        assert await person.image(image) == PREDICTION
        assert await person.image(b"png bytes", content_type="image/png") == PREDICTION
        assert await person.image("https://example.com/1.jpg") == PREDICTION
        assert await person.image(str(image)) == PREDICTION

    by_url, by_asset, by_file, by_bytes, by_str_url, by_str_path = pipeline_api.requests
    pop = {"components": [{"type": "inference", "ability": "eyepop.person"}]}
    assert by_url.content_type == "application/json"
    assert by_url.json == {"pop": pop, "source": {"url": "https://example.com/1.jpg"}}
    assert by_asset.json == {"pop": {"components": []}, "source": {"assetUuid": "4d1e"}}
    assert by_file.content_type == "multipart/form-data"
    assert by_file.parts["request"] == ("application/json", b'{"pop": {"components": [{"type": "inference", "ability": "eyepop.person"}]}}')
    assert by_file.parts["image"] == ("image/jpeg", b"\xff\xd8 jpeg bytes")
    assert by_bytes.parts["image"] == ("image/png", b"png bytes")
    assert by_str_url.json == by_url.json
    assert by_str_path.parts == by_file.parts
    assert all(r.authorization == f"Bearer {JWT}" for r in pipeline_api.requests)


def test_pop_json():
    """An ability or a model is a pop of one inference component; crop forwards to its targets; a Pop is sent as written."""
    assert Model("model-uuid").json == {"components": [{"type": "inference", "modelUuid": "model-uuid"}]}
    assert Pop({"components": []}).json == {"components": []}
    pose = PERSON.crop(Ability("eyepop.person.2d-body-points"), Model("m").crop(Ability("eyepop.text")))
    assert pose.json == {
        "components": [
            {
                "type": "inference",
                "ability": "eyepop.person",
                "forward": {
                    "operator": {"type": "crop"},
                    "targets": [
                        {"type": "inference", "ability": "eyepop.person.2d-body-points"},
                        {
                            "type": "inference",
                            "modelUuid": "m",
                            "forward": {"operator": {"type": "crop"}, "targets": [{"type": "inference", "ability": "eyepop.text"}]},
                        },
                    ],
                },
            }
        ]
    }
    assert PERSON.json == {"components": [{"type": "inference", "ability": "eyepop.person"}]}  # crop returned a copy


# Retry and errors


@pytest.mark.parametrize("error", ["overloaded", "pipeline_stopped"])
async def test_image_retries_a_503_with_retry_after(pipeline_api: FakePipelineAPI, error: str):
    """A 503 with Retry-After is retried, and the 200 that follows is returned."""
    pipeline_api.post_answers.append((503, {"error": error, "message": "busy"}, {"Retry-After": "1"}))
    async with EyePop(pipeline_api.url, access_token=JWT) as client:
        assert await client.use(PERSON).image({"url": "https://example.com/1.jpg"}) == PREDICTION
    assert len(pipeline_api.requests) == 2


async def test_image_raises_the_last_503_once_the_bound_has_passed(pipeline_api: FakePipelineAPI):
    """A 503 still answered once the retry bound has passed is raised as it arrived."""
    overloaded = (503, {"error": "overloaded", "retry_after_seconds": 1}, {"Retry-After": "1"})
    pipeline_api.post_answers.extend([overloaded] * 5)
    async with EyePop(pipeline_api.url, access_token=JWT, wait=1.5) as client:
        with pytest.raises(EyePopError) as raised:
            await client.use(PERSON).image({"url": "https://example.com/1.jpg"})
    assert (raised.value.status, raised.value.error, raised.value.close_code) == (503, "overloaded", None)
    assert len(pipeline_api.requests) == 2


async def test_image_waits_at_least_a_second_on_a_zero_retry_after(pipeline_api: FakePipelineAPI):
    """A 503 with Retry-After 0 is retried after a second, not at once."""
    pipeline_api.post_answers.append((503, {"error": "overloaded", "message": "busy"}, {"Retry-After": "0"}))
    async with EyePop(pipeline_api.url, access_token=JWT) as client:
        started = time.monotonic()
        assert await client.use(PERSON).image({"url": "https://example.com/1.jpg"}) == PREDICTION
    assert time.monotonic() - started >= 1.0
    assert len(pipeline_api.requests) == 2


@pytest.mark.parametrize("body", ["<html>proxy page</html>", {"no": "predictions"}, {"predictions": []}, ["a list"]])
async def test_image_raises_a_200_with_no_predictions(pipeline_api: FakePipelineAPI, body: Any):
    """A 200 that is not JSON or holds no prediction raises with status 200."""
    pipeline_api.post_answers.append((200, body, {}))
    async with EyePop(pipeline_api.url, access_token=JWT) as client:
        with pytest.raises(EyePopError) as raised:
            await client.use(PERSON).image({"url": "https://example.com/1.jpg"})
    assert (raised.value.status, raised.value.close_code) == (200, None)


@pytest.mark.parametrize("retry_after", ["-1", "nan"])
async def test_image_raises_a_503_with_an_invalid_retry_after_at_once(pipeline_api: FakePipelineAPI, retry_after: str):
    """A 503 whose Retry-After is negative or not a number raises at once."""
    pipeline_api.post_answers.append((503, {"error": "overloaded", "message": "busy"}, {"Retry-After": retry_after}))
    async with EyePop(pipeline_api.url, access_token=JWT) as client:
        with pytest.raises(EyePopError) as raised:
            await client.use(PERSON).image({"url": "https://example.com/1.jpg"})
    assert (raised.value.status, raised.value.error) == (503, "overloaded")
    assert len(pipeline_api.requests) == 1


async def test_image_raises_other_errors_at_once(pipeline_api: FakePipelineAPI):
    """A 422 raises at once with its status, error and message."""
    pipeline_api.post_answers.append((422, {"error": "invalid_pop", "message": "no components"}, {}))
    async with EyePop(pipeline_api.url, access_token=JWT) as client:
        with pytest.raises(EyePopError) as raised:
            await client.use(PERSON).image({"url": "https://example.com/1.jpg"})
    assert (raised.value.status, raised.value.error, raised.value.message) == (422, "invalid_pop", "no components")
    assert len(pipeline_api.requests) == 1


# An API key


async def test_api_key_is_exchanged_and_never_sent_to_the_pipeline_api(pipeline_api: FakePipelineAPI, compute_api: FakeComputeAPI):
    """The API key reaches only compute-api; the pipeline API sees only the minted token, exchanged once while it is valid."""
    async with EyePop(pipeline_api.url, api_key=API_KEY, compute_url=compute_api.url) as client:
        await client.use(PERSON).image({"url": "https://example.com/1.jpg"})
        await collect(client)
    assert compute_api.authorizations == [f"Bearer {API_KEY}"]  # the token has an hour left: one exchange
    assert [r.authorization for r in pipeline_api.requests] == ["Bearer minted-1"] * 2


async def test_a_request_in_the_tokens_last_minute_exchanges_first(pipeline_api: FakePipelineAPI, compute_api: FakeComputeAPI):
    """A request in the token's last 60 s is preceded by a second exchange."""
    compute_api.expires_in = 30
    async with EyePop(pipeline_api.url, api_key=API_KEY, compute_url=compute_api.url) as client:
        await client.use(PERSON).image({"url": "https://example.com/1.jpg"})
        await collect(client)
    assert len(compute_api.authorizations) == 2
    assert [r.authorization for r in pipeline_api.requests] == ["Bearer minted-1", "Bearer minted-2"]


async def test_callers_in_the_last_minute_share_one_exchange(pipeline_api: FakePipelineAPI, compute_api: FakeComputeAPI):
    """Callers that arrive together in the token's last minute share one exchange, even when the new token is as short."""
    compute_api.expires_in = 30
    async with EyePop(pipeline_api.url, api_key=API_KEY, compute_url=compute_api.url) as client:
        assert await client._valid_token() == "minted-1"  # pyright: ignore[reportPrivateUsage]
        tokens = await asyncio.gather(*(client._valid_token() for _ in range(6)))  # pyright: ignore[reportPrivateUsage]
    assert tokens == ["minted-2"] * 6
    assert len(compute_api.authorizations) == 2


async def test_a_401_exchanges_again_and_retries_once(pipeline_api: FakePipelineAPI, compute_api: FakeComputeAPI):
    """A POST or an upgrade answered 401 is retried once with a newly exchanged token."""
    pipeline_api.post_answers.append((401, "Unauthorized", {}))
    pipeline_api.upgrade_answers.append((401, "Unauthorized", {}))
    async with EyePop(pipeline_api.url, api_key=API_KEY, compute_url=compute_api.url) as client:
        assert await client.use(PERSON).image({"url": "https://example.com/1.jpg"}) == PREDICTION
        await collect(client)
    assert [r.authorization for r in pipeline_api.requests] == [f"Bearer minted-{n}" for n in (1, 2, 2, 3)]
    assert len(compute_api.authorizations) == 3


async def test_a_second_401_raises(pipeline_api: FakePipelineAPI, compute_api: FakeComputeAPI):
    """A 401 answered again after the new exchange raises, with no further exchange."""
    pipeline_api.post_answers.extend([(401, "Unauthorized", {})] * 3)
    async with EyePop(pipeline_api.url, api_key=API_KEY, compute_url=compute_api.url) as client:
        with pytest.raises(EyePopError) as raised:
            await client.use(PERSON).image({"url": "https://example.com/1.jpg"})
    assert raised.value.status == 401
    assert len(pipeline_api.requests) == 2
    assert len(compute_api.authorizations) == 2


async def test_401s_that_arrive_together_share_one_exchange(pipeline_api: FakePipelineAPI, compute_api: FakeComputeAPI):
    """Requests rejected with the same token share one new exchange."""
    pipeline_api.post_answers.extend([(401, "Unauthorized", {})] * 3)
    async with EyePop(pipeline_api.url, api_key=API_KEY, compute_url=compute_api.url) as client:
        await client._valid_token()  # pyright: ignore[reportPrivateUsage]
        images = [client.use(PERSON).image({"url": "https://example.com/1.jpg"}) for _ in range(3)]
        assert await asyncio.gather(*images) == [PREDICTION] * 3
    assert len(compute_api.authorizations) == 2


async def test_a_401_with_an_access_token_raises_at_once(pipeline_api: FakePipelineAPI):
    """With an access token there is nothing to exchange: a 401 raises at once."""
    pipeline_api.post_answers.append((401, "Unauthorized", {}))
    async with EyePop(pipeline_api.url, access_token=JWT) as client:
        with pytest.raises(EyePopError) as raised:
            await client.use(PERSON).image({"url": "https://example.com/1.jpg"})
    assert raised.value.status == 401
    assert len(pipeline_api.requests) == 1


async def test_a_failed_exchange_raises_and_sends_nothing(pipeline_api: FakePipelineAPI, compute_api: FakeComputeAPI):
    """A refused exchange raises with compute-api's status, names compute-api, and sends the pipeline API nothing."""
    compute_api.status = 401
    async with EyePop(pipeline_api.url, api_key=API_KEY, compute_url=compute_api.url) as client:
        with pytest.raises(EyePopError) as by_image:
            await client.use(PERSON).image({"url": "https://example.com/1.jpg"})
        with pytest.raises(EyePopError) as by_socket:
            await collect(client)
    for raised in (by_image, by_socket):
        assert (raised.value.status, raised.value.error, raised.value.close_code) == (401, None, None)
        assert "compute-api" in raised.value.message
    assert pipeline_api.requests == []


async def test_an_exchange_answer_with_no_valid_expiry_raises_and_sends_nothing(pipeline_api: FakePipelineAPI, compute_api: FakeComputeAPI):
    """An exchange answered with a null expires_in raises and sends the pipeline API nothing."""
    compute_api.expires_in = None
    async with EyePop(pipeline_api.url, api_key=API_KEY, compute_url=compute_api.url) as client:
        with pytest.raises(EyePopError) as raised:
            await client.use(PERSON).image({"url": "https://example.com/1.jpg"})
    assert (raised.value.status, raised.value.close_code) == (None, None)
    assert pipeline_api.requests == []


# Plain http off loopback


@pytest.mark.parametrize(
    "kwargs",
    [
        {"url": "http://pipeline.example.com"},
        {"url": "http://10.0.0.5:8080"},
        {"url": "https://pipeline.example.com", "compute_url": "http://compute.example.com"},
    ],
)
def test_plain_http_off_loopback_is_refused(kwargs: dict[str, str]):
    """A plain http url or compute url off loopback is refused before any request."""
    with pytest.raises(ValueError):
        EyePop(**kwargs, api_key=API_KEY)


@pytest.mark.parametrize("url", ["http://localhost:8080", "http://127.0.0.1", "http://[::1]:8080", "https://pipeline.example.com"])
def test_loopback_and_https_are_accepted(url: str):
    """https, and plain http to a loopback host, are accepted."""
    EyePop(url, access_token=JWT, compute_url="http://localhost:9000")


@pytest.mark.parametrize("kwargs", [{}, {"access_token": JWT, "api_key": API_KEY}])
def test_exactly_one_credential(kwargs: dict[str, str]):
    """Neither credential, or both, is refused."""
    with pytest.raises(ValueError):
        EyePop("https://pipeline.example.com", **kwargs)


# No credential in errors or logs


async def test_no_credential_in_errors_or_logs(pipeline_api: FakePipelineAPI, compute_api: FakeComputeAPI, caplog: pytest.LogCaptureFixture):
    """No error message or log record holds the API key or a token."""
    caplog.set_level(logging.DEBUG)
    pipeline_api.post_answers.extend([(401, "Unauthorized", {})] * 2)
    pipeline_api.script = closes(1008, b"credential_expired")
    errors: list[EyePopError] = []
    async with EyePop(pipeline_api.url, api_key=API_KEY, compute_url=compute_api.url) as client:
        with pytest.raises(EyePopError) as raised:
            await client.use(PERSON).image({"url": "https://example.com/1.jpg"})
        errors.append(raised.value)
        assert (raised.value.status, raised.value.error, raised.value.message) == (401, None, "Unauthorized")
        with pytest.raises(EyePopError) as raised:
            await collect(client)
        errors.append(raised.value)
    compute_api.status = 403
    async with EyePop(pipeline_api.url, api_key=API_KEY, compute_url=compute_api.url) as client:
        with pytest.raises(EyePopError) as raised:
            await client.use(PERSON).image({"url": "https://example.com/1.jpg"})
        errors.append(raised.value)
    secrets = [API_KEY, *compute_api.minted]
    texts = [str(e) for e in errors] + [e.message for e in errors] + [r.getMessage() for r in caplog.records]
    assert not [t for t in texts if any(s in t for s in secrets)]


# An isolated module tree


def test_pipeline_and_auth_import_nothing_else_from_the_sdk():
    """eyepop/pipeline and eyepop/auth import no eyepop module outside those two trees."""
    root = pathlib.Path(eyepop.pipeline.__file__).parents[1]
    allowed = ("eyepop.pipeline", "eyepop.auth")
    outside = []
    for tree in ("pipeline", "auth"):
        files = list((root / tree).rglob("*.py"))
        assert files
        for path in files:
            for node in ast.walk(ast.parse(path.read_text())):
                if isinstance(node, ast.Import):
                    names = [alias.name for alias in node.names]
                elif isinstance(node, ast.ImportFrom):
                    if node.level > 1:
                        outside.append(f"{path}: relative import beyond its package")
                    names = [node.module or ""] if node.level == 0 else []
                else:
                    continue
                for name in names:
                    if (name == "eyepop" or name.startswith("eyepop.")) and not name.startswith(allowed):
                        outside.append(f"{path}: {name}")
    assert outside == []


# Dead connections


async def test_a_socket_with_no_pong_raises_1006_within_one_and_a_half_heartbeats():
    """A socket that answers no ping raises 1006 within 1.5 heartbeats, and the pipeline's pop never gets its answer."""
    async def silent(ws: web.WebSocketResponse, fake: FakePipelineAPI) -> None:
        await asyncio.sleep(10)

    heartbeat = 0.4
    async with FakePipelineAPI(script=silent, autoping=False) as fake, EyePop(fake.url, access_token=JWT, heartbeat=heartbeat) as client:
        started = time.monotonic()
        with pytest.raises(EyePopError) as raised:
            async with client.pipeline(PERSON):
                pass
        elapsed = time.monotonic() - started
    assert (raised.value.status, raised.value.close_code) == (None, 1006)
    assert elapsed <= 1.5 * heartbeat + 0.2  # scheduler jitter on loaded runners


async def test_a_connection_dropped_with_no_close_frame_raises_1006():
    """A connection dropped with no close frame raises 1006."""
    async def drop(ws: web.WebSocketResponse, fake: FakePipelineAPI) -> None:
        transport = ws._req.transport if ws._req else None  # pyright: ignore[reportPrivateUsage]
        assert transport is not None
        transport.close()

    async with FakePipelineAPI(script=drop) as fake, EyePop(fake.url, access_token=JWT) as client:
        with pytest.raises(EyePopError) as raised:
            await collect(client)
    assert (raised.value.status, raised.value.close_code) == (None, 1006)


async def test_nothing_listening_raises_with_no_status_and_no_close_code():
    """A port nothing listens on raises with no status and no close code, the library's exception as cause."""
    async with EyePop(f"http://127.0.0.1:{free_port()}", access_token=JWT) as client:
        with pytest.raises(EyePopError) as by_image:
            await client.use(PERSON).image({"url": "https://example.com/1.jpg"})
        with pytest.raises(EyePopError) as by_socket:
            await collect(client)
    for raised in (by_image, by_socket):
        assert (raised.value.status, raised.value.close_code, raised.value.error) == (None, None, None)
        assert raised.value.message
        assert raised.value.__cause__ is not None


async def test_compute_api_down_raises_an_exchange_error(pipeline_api: FakePipelineAPI):
    """compute-api down raises with no status, names compute-api, and sends the pipeline API nothing."""
    async with EyePop(pipeline_api.url, api_key=API_KEY, compute_url=f"http://127.0.0.1:{free_port()}") as client:
        with pytest.raises(EyePopError) as raised:
            await client.use(PERSON).image({"url": "https://example.com/1.jpg"})
    assert (raised.value.status, raised.value.close_code) == (None, None)
    assert "compute-api" in raised.value.message
    assert API_KEY not in str(raised.value)
    assert pipeline_api.requests == []


# client()


@pytest.fixture
def dotenv(tmp_path: pathlib.Path, monkeypatch: pytest.MonkeyPatch) -> pathlib.Path:
    monkeypatch.chdir(tmp_path)
    for name in ("EYEPOP_API_KEY", "EYEPOP_ACCESS_TOKEN", "EYEPOP_PIPELINE_URL"):
        monkeypatch.delenv(name, raising=False)
    return tmp_path / ".env"


async def test_init_reads_the_environment_over_dotenv(dotenv: pathlib.Path, monkeypatch: pytest.MonkeyPatch, pipeline_api: FakePipelineAPI, compute_api: FakeComputeAPI):
    """`client()` takes the url and the API key from .env, the environment winning over it, and exchanges the key it read."""
    dotenv.write_text(f'# a comment\nexport EYEPOP_PIPELINE_URL="{pipeline_api.url}"\nEYEPOP_API_KEY=dotenv-key\n')
    monkeypatch.setenv("EYEPOP_API_KEY", API_KEY)
    async with eyepop.pipeline.client(compute_url=compute_api.url) as client:
        assert await client.use(PERSON).image("https://example.com/1.jpg") == PREDICTION
    assert compute_api.authorizations == [f"Bearer {API_KEY}"]


async def test_init_falls_back_to_an_access_token(dotenv: pathlib.Path, pipeline_api: FakePipelineAPI):
    """With no API key, `client()` sends EYEPOP_ACCESS_TOKEN as it is."""
    dotenv.write_text(f"EYEPOP_PIPELINE_URL={pipeline_api.url}\nEYEPOP_ACCESS_TOKEN={JWT}\n")
    async with eyepop.pipeline.client() as client:
        await client.use(PERSON).image("https://example.com/1.jpg")
    assert pipeline_api.requests[0].authorization == f"Bearer {JWT}"


@pytest.mark.parametrize("env", ["", "EYEPOP_API_KEY=k\n", "EYEPOP_PIPELINE_URL=https://pipeline.example.com\n"])
def test_init_needs_a_url_and_a_credential(dotenv: pathlib.Path, env: str):
    """With no url or no credential anywhere, `client()` refuses."""
    dotenv.write_text(env)
    with pytest.raises(ValueError):
        eyepop.pipeline.client()


async def test_configure_is_what_client_falls_back_to(dotenv: pathlib.Path, pipeline_api: FakePipelineAPI):
    """`configure()` supplies the url and the token when nothing is passed, and what is passed wins."""
    eyepop.pipeline.configure(url=pipeline_api.url, access_token=JWT)
    try:
        async with eyepop.pipeline.client() as client:
            await client.use(PERSON).image("https://example.com/1.jpg")
        async with eyepop.pipeline.client(access_token="other.jwt.token") as client:
            await client.use(PERSON).image("https://example.com/1.jpg")
    finally:
        eyepop.pipeline.configure(url=None, access_token=None)
    assert [r.authorization for r in pipeline_api.requests] == [f"Bearer {JWT}", "Bearer other.jwt.token"]
