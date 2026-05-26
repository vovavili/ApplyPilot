"""Small Playwright helpers shared by discovery and enrichment."""

from __future__ import annotations

import logging
from typing import Any

from applypilot.config import get_chrome_path

log = logging.getLogger(__name__)


def launch_chromium(chromium: Any, **launch_options: Any) -> Any:
    """Launch Playwright Chromium, falling back to installed Chrome when needed."""
    try:
        return chromium.launch(**launch_options)
    except Exception as exc:
        message = str(exc).casefold()
        if "executable doesn't exist" not in message and "playwright install" not in message:
            raise

        chrome_path = get_chrome_path()
        log.warning("Playwright browser binary is missing; falling back to Chrome at %s", chrome_path)
        return chromium.launch(**{**launch_options, "executable_path": chrome_path})
