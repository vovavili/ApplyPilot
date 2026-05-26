"""Opt-in humanizer prompt injection for generated job-application prose."""

from __future__ import annotations

import os
from pathlib import Path

from applypilot.config import APP_DIR, load_env

DEFAULT_PROMPT_PATH = APP_DIR / "humanizer.md"
DEFAULT_MAX_PROMPT_CHARS = 4500
TRUE_VALUES = {"1", "true", "yes", "on"}

FALLBACK_PROMPT = """
Use this as a final pass for generated job-application prose.

Preserve facts:
- Do not invent skills, metrics, employers, degrees, certifications, projects, or work history.
- Do not change real numbers. If a fact is not in the resume, profile, or job description, leave it out.
- Keep job-specific keywords where they are accurate.

Make the writing sound human:
- Replace generic AI phrasing with plain, specific language.
- Avoid corporate filler such as "aligns with", "passionate about", "excited to apply",
  "leveraged cutting-edge", "dynamic", "innovative", "pivotal", "showcase", and "underscore".
- Avoid vague claims. Prefer concrete work, tools, constraints, outcomes, and trade-offs.
- Vary sentence rhythm, but keep the text professional and easy to scan.
- Do not over-polish. A little directness is better than a perfect brochure voice.

Before returning the final text, silently ask: "What makes this sound obviously AI-generated?"
Then revise those parts once.
""".strip()

TARGET_NOTES = {
    "resume": """
Resume-specific rules:
- Apply the humanizer only to generated summary and bullet strings.
- Keep the resume terse, ATS-scannable, and keyword-aware.
- Do not use first person.
- Do not make bullets chatty or clever. Strong, concrete, boringly useful is the target.
""".strip(),
    "cover_letter": """
Cover-letter-specific rules:
- A measured first-person voice is fine.
- Sound like a capable person writing to another person, not a template.
- Keep the existing structure and word limit.
""".strip(),
    "screening": """
Screening-answer-specific rules:
- Apply this only to open-ended prose answers.
- For salary or compensation fields, follow the SALARY STRATEGY section. Never reveal a private minimum.
- For factual, legal, work-authorization, demographic, or yes/no fields, use the profile exactly.
- Keep answers short unless the form asks for detail.
""".strip(),
}


def get_humanizer_prompt(target: str) -> str:
    """Return the humanizer prompt section for a target, or an empty string.

    Set APPLYPILOT_HUMANIZER=1 to enable. By default the prompt is loaded from
    ~/.applypilot/humanizer.md. Override with APPLYPILOT_HUMANIZER_PROMPT_PATH.
    """
    load_env()
    if os.environ.get("APPLYPILOT_HUMANIZER", "").strip().lower() not in TRUE_VALUES:
        return ""

    target_note = TARGET_NOTES.get(target, "").strip()
    prompt = _load_humanizer_text()

    return f"""

## HUMANIZER PASS
Apply these rules only after satisfying all task-specific rules above. If there is a conflict, factual accuracy, validation rules, output format, and application safety rules win.

{target_note}

{prompt}
""".rstrip()


def _load_humanizer_text() -> str:
    path = _configured_prompt_path()
    if path.exists():
        text = path.read_text(encoding="utf-8").strip()
        if text:
            return text[: _max_prompt_chars()].rstrip()
    return FALLBACK_PROMPT


def _configured_prompt_path() -> Path:
    raw_path = os.environ.get("APPLYPILOT_HUMANIZER_PROMPT_PATH", "").strip()
    if raw_path:
        return Path(raw_path).expanduser()
    return DEFAULT_PROMPT_PATH


def _max_prompt_chars() -> int:
    raw_value = os.environ.get("APPLYPILOT_HUMANIZER_MAX_CHARS", "")
    if not raw_value:
        return DEFAULT_MAX_PROMPT_CHARS
    try:
        return max(500, int(raw_value))
    except ValueError:
        return DEFAULT_MAX_PROMPT_CHARS
