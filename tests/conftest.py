import pytest
import requests


@pytest.fixture(autouse=True)
def no_network(monkeypatch):
    """Any real HTTP request made through `requests` fails inside the test suite.

    The LLM providers and the downloader all use `requests`, so tests can never call a real
    API or need keys; they use fake sessions and the FakeProvider instead.
    """

    def blocked(self, request, *args, **kwargs):
        raise RuntimeError(f"network access blocked in tests: {request.method} {request.url}")

    monkeypatch.setattr(requests.adapters.HTTPAdapter, "send", blocked)
