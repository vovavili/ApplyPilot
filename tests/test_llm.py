import httpx
import pytest

from applypilot.llm import AnthropicLLMClient


class _FakeHTTPClient:
    def __init__(self, response: httpx.Response) -> None:
        self.response = response
        self.payloads: list[dict] = []
        self.headers: list[dict] = []

    def post(self, _url, *, headers, json):
        self.headers.append(headers)
        self.payloads.append(json)
        return self.response


def test_anthropic_client_omits_temperature_for_newer_claude_models():
    response = httpx.Response(
        200,
        json={"content": [{"type": "text", "text": '{"ok": true}'}]},
        request=httpx.Request("POST", "https://api.anthropic.com/v1/messages"),
    )
    http_client = _FakeHTTPClient(response)
    client = AnthropicLLMClient("claude-opus-4-7", "test-key")
    client._client = http_client

    result = client.chat(
        [
            {"role": "system", "content": "Return JSON."},
            {"role": "user", "content": "Hello"},
        ],
        temperature=0.2,
        max_tokens=200,
    )

    assert result == '{"ok": true}'
    payload = http_client.payloads[0]
    assert payload["model"] == "claude-opus-4-7"
    assert payload["system"] == "Return JSON."
    assert payload["messages"] == [{"role": "user", "content": "Hello"}]
    assert "temperature" not in payload


def test_anthropic_client_reports_error_body():
    response = httpx.Response(
        400,
        json={"error": {"message": "`temperature` is deprecated for this model."}},
        request=httpx.Request("POST", "https://api.anthropic.com/v1/messages"),
    )
    client = AnthropicLLMClient("claude-opus-4-7", "test-key")
    client._client = _FakeHTTPClient(response)

    with pytest.raises(RuntimeError, match="temperature.*deprecated"):
        client.chat([{"role": "user", "content": "Hello"}])
