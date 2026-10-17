from dataclasses import dataclass

import aiohttp


class ExchangeError(Exception):
    """`status` is None when compute-api could not be reached. The message never holds a credential."""

    def __init__(self, status: int | None, message: str):
        super().__init__(message)
        self.status = status
        self.message = message


@dataclass(frozen=True)
class Token:
    access_token: str
    expires_in: int
    expires_at: str


async def exchange(session: aiohttp.ClientSession, compute_url: str, api_key: str) -> Token:
    headers = {"Authorization": f"Bearer {api_key}", "Accept": "application/json"}
    try:
        async with session.post(f"{compute_url}/v1/auth/authenticate", headers=headers) as response:
            if response.status >= 400:
                raise ExchangeError(
                    response.status,
                    f"compute-api refused the API key exchange: HTTP {response.status} {response.reason or ''}".rstrip(),
                )
            body = await response.json()
    except (aiohttp.ClientError, OSError, TimeoutError, ValueError) as e:
        raise ExchangeError(None, f"could not exchange the API key at compute-api: {type(e).__name__}: {e}") from e
    if not isinstance(body, dict):
        raise ExchangeError(response.status, "compute-api answered the API key exchange with no JSON object")
    return Token(
        access_token=body.get("access_token", ""),
        expires_in=body.get("expires_in", 0),
        expires_at=body.get("expires_at", ""),
    )
