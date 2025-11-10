import pytest
from aioresponses import aioresponses as aioresponses_mocker


@pytest.fixture
def aioresponses():
    """Fixture to mock aiohttp requests."""
    with aioresponses_mocker() as m:
        yield m
