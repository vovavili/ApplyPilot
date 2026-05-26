import pytest

from applypilot import playwright_utils


class FakeChromium:
    def __init__(self, error_message: str):
        self.error_message = error_message
        self.calls = []

    def launch(self, **options):
        self.calls.append(options)
        if len(self.calls) == 1:
            raise RuntimeError(self.error_message)
        return {"browser": "ok", "options": options}


def test_launch_chromium_falls_back_to_system_chrome_when_browser_binary_is_missing(monkeypatch):
    chromium = FakeChromium("Executable doesn't exist. Please run playwright install")
    monkeypatch.setattr(playwright_utils, "get_chrome_path", lambda: "C:/Program Files/Google/Chrome/chrome.exe")

    browser = playwright_utils.launch_chromium(chromium, headless=True)

    assert browser["browser"] == "ok"
    assert chromium.calls == [
        {"headless": True},
        {"headless": True, "executable_path": "C:/Program Files/Google/Chrome/chrome.exe"},
    ]


def test_launch_chromium_does_not_hide_unrelated_launch_errors():
    chromium = FakeChromium("proxy authentication failed")

    with pytest.raises(RuntimeError, match="proxy authentication failed"):
        playwright_utils.launch_chromium(chromium, headless=True)
