"""Cover letter generation with structured LLM output and deterministic assembly."""

import json
import logging
import os
import re
import time
from datetime import datetime, timezone
from hashlib import sha1
from pathlib import Path

from applypilot.config import COVER_LETTER_DIR, RESUME_PATH, load_profile, load_search_config
from applypilot.database import get_connection
from applypilot.discovery import workday
from applypilot.events import emit_error, emit_event
from applypilot.llm import get_client, make_anthropic_client, make_client
from applypilot.scoring.validator import sanitize_text, validate_cover_letter

log = logging.getLogger(__name__)

MAX_ATTEMPTS = 5  # max cross-run retries before giving up

COVER_JSON_FIELDS = (
    "opening_work",
    "role_fit",
    "achievement_1",
    "achievement_2",
    "company_reason",
    "closing",
)

COVER_MIN_WORDS = 170
COVER_TARGET_MAX_WORDS = 285
COVER_HARD_MAX_WORDS = 340

JOB_KEYWORDS = (
    "Python",
    "SQL",
    "Databricks",
    "Azure",
    "Microsoft Fabric",
    "Power BI",
    "Apache Airflow",
    "dbt",
    "Spark",
    "Docker",
    "PostgreSQL",
    "Kafka",
    "Snowflake",
    "BigQuery",
    "Cosmos DB",
    "Pydantic",
    "Great Expectations",
    "ETL",
    "ELT",
    "data quality",
    "reporting",
    "analytics",
)

BROAD_JOB_BOARD_SITES = {
    "indeed",
    "linkedin",
    "glassdoor",
    "zip_recruiter",
    "ziprecruiter",
}


# -- Prompt Builder ---------------------------------------------------------


def _flatten_allowed_skills(profile: dict) -> list[str]:
    skills: list[str] = []
    for items in profile.get("skills_boundary", {}).values():
        if isinstance(items, list):
            skills.extend(str(item).strip() for item in items if str(item).strip())
    return list(dict.fromkeys(skills))


def _resume_fact_lines(profile: dict) -> list[str]:
    facts = profile.get("resume_facts", {})
    lines: list[str] = []
    for key, value in facts.items():
        if isinstance(value, list):
            clean = [str(item).strip() for item in value if str(item).strip()]
            if clean:
                lines.append(f"- {key}: {', '.join(clean[:8])}")
        elif value:
            lines.append(f"- {key}: {value}")
    return lines[:16]


def _build_cover_letter_prompt(profile: dict) -> str:
    """Build a structured JSON cover-letter prompt from the user's profile."""
    personal = profile.get("personal", {})
    sign_off_name = personal.get("preferred_name") or personal.get("full_name", "")
    skills = _flatten_allowed_skills(profile)
    skills_str = ", ".join(skills) if skills else "the tools listed in the supplied resume"
    fact_lines = _resume_fact_lines(profile)
    fact_block = "\n".join(fact_lines) if fact_lines else "- Use only facts present in the supplied resume."

    fields = "\n".join(
        (
            '- "opening_work": one or two sentences tying the target role to concrete data work',
            '- "role_fit": one sentence connecting the role requirements to the candidate stack',
            '- "achievement_1": one sentence with a specific resume fact and tools or domain',
            '- "achievement_2": one sentence with a second specific resume fact and tools or domain',
            '- "company_reason": one or two sentences about the company, team, product, or problem from the job post',
            '- "closing": one sentence asking for a conversation about the work',
        )
    )
    return f"""You write cover-letter building blocks as structured JSON.

Return ONLY valid JSON. No markdown fences, no prose before the JSON, no notes after it.

Required JSON fields:
{fields}

The code will add the greeting, paragraphs, and signoff. Do not include "Dear Hiring Manager" or the candidate name.

Voice rules:
- Write like a working engineer, direct and concrete.
- Use only facts from the supplied resume and target job.
- Prefer specific work, tools, outcomes, teams, and domains over generic enthusiasm.
- The final assembled letter should be about {COVER_MIN_WORDS}-{COVER_TARGET_MAX_WORDS} words.
- Keep fields detailed enough to support a real letter, but do not ramble.
- Write a single JSON object ending with the closing brace.
- No em dash or en dash. Do not start fields with "I am writing" or "I am excited".
- Do not claim experience with tools outside the allowed list unless that tool appears in the supplied resume.
- If the target asks for an unfamiliar tool, describe nearby work instead of pretending.

Candidate signoff name: {sign_off_name}
Allowed candidate tools: {skills_str}
Known resume facts:
{fact_block}
"""


# -- Helpers ----------------------------------------------------------------


def _cover_model_override() -> str:
    return os.environ.get("COVER_LLM_MODEL", "").strip()


def _cover_provider_override() -> str:
    return os.environ.get("COVER_LLM_PROVIDER", "").strip().casefold()


def _env_int(name: str, default: int, minimum: int = 1) -> int:
    value = os.environ.get(name, "").strip()
    if not value:
        return default
    try:
        return max(int(value), minimum)
    except ValueError:
        log.warning("Invalid %s=%r; using %s", name, value, default)
        return default


def _cover_max_retries() -> int:
    return _env_int("COVER_MAX_RETRIES", 3, minimum=0)


def _cover_max_tokens() -> int:
    return _env_int("COVER_LLM_MAX_TOKENS", 2200, minimum=1000)


def _is_claude_model(model: str) -> bool:
    normalized = model.strip().casefold()
    return normalized.startswith(("claude-", "anthropic/claude-"))


def _get_cover_client():
    provider = _cover_provider_override()
    model = _cover_model_override()
    if provider in {"anthropic", "claude"} or (model and _is_claude_model(model)):
        return make_anthropic_client(model=model or None)
    if model:
        return make_client(model=model)
    return get_client()


def _extract_cover_json(raw: str) -> dict:
    """Extract JSON from an LLM response, tolerating fences or a short preamble."""
    raw = raw.strip()
    try:
        parsed = json.loads(raw)
    except json.JSONDecodeError:
        parsed = None
    if isinstance(parsed, dict):
        return parsed

    if "```" in raw:
        for part in raw.split("```")[1::2]:
            part = part.strip()
            if part.startswith("json"):
                part = part[4:].strip()
            try:
                parsed = json.loads(part)
            except json.JSONDecodeError:
                continue
            if isinstance(parsed, dict):
                return parsed

    start = raw.find("{")
    end = raw.rfind("}")
    if start != -1 and end > start:
        try:
            parsed = json.loads(raw[start : end + 1])
        except json.JSONDecodeError:
            parsed = None
        if isinstance(parsed, dict):
            return parsed

    parsed = _extract_json_like_fields(raw)
    if parsed:
        return parsed

    raise ValueError("No valid JSON object found in LLM response")


def _extract_json_like_fields(raw: str) -> dict | None:
    """Recover complete fields from almost-JSON output with a missing brace."""
    recovered: dict[str, str] = {}
    for field in COVER_JSON_FIELDS:
        pattern = re.compile(
            rf'(?:["`])?{re.escape(field)}(?:["`])?\s*:\s*"(?P<value>(?:\\.|[^"\\])*)"',
            re.IGNORECASE | re.DOTALL,
        )
        match = pattern.search(raw)
        if not match:
            return None
        value = match.group("value")
        try:
            recovered[field] = json.loads(f'"{value}"')
        except json.JSONDecodeError:
            recovered[field] = value.replace('\\"', '"').replace("\\n", " ")
    return recovered


def _clean_sentence(value: object, field: str) -> str:
    text = sanitize_text(str(value or "")).strip()
    text = re.sub(r"\s+", " ", text)
    text = text.strip(" -*\t\r\n")
    if not text:
        raise ValueError(f"Missing structured field: {field}")
    if text.lower().startswith("dear hiring manager"):
        raise ValueError(f"Field must not include greeting: {field}")
    if text[-1] not in ".!?":
        text += "."
    return text


def _display_title(job: dict) -> str:
    title = str(job.get("title") or "this role").strip() or "this role"
    site = str(job.get("site") or "").casefold()
    if site in BROAD_JOB_BOARD_SITES and "|" in title:
        title = title.rsplit("|", 1)[0].strip() or title
    return title


def _display_company(job: dict) -> str:
    company = str(job.get("company") or "").strip()
    if company:
        return company

    title = str(job.get("title") or "").strip()
    site = str(job.get("site") or "").strip()
    if site.casefold() in BROAD_JOB_BOARD_SITES:
        if "|" in title:
            inferred = title.rsplit("|", 1)[1].strip()
            if inferred:
                return inferred
        return "the hiring team"
    return site or "the hiring team"


def _job_context_sentence(job: dict) -> str:
    title = _display_title(job)
    company = _display_company(job)
    return (
        f"For the {title} role at {company}, the relevant overlap is practical data work: "
        "ingestion, modeling, validation, reporting, and production ownership."
    )


def _assemble_cover_letter(data: dict, job: dict, profile: dict) -> str:
    """Assemble a complete cover letter from structured model fields."""
    missing = [field for field in COVER_JSON_FIELDS if not str(data.get(field) or "").strip()]
    if missing:
        raise ValueError(f"Missing structured fields: {', '.join(missing)}")

    personal = profile.get("personal", {})
    sign_off_name = personal.get("preferred_name") or personal.get("full_name", "")
    if not sign_off_name:
        raise ValueError("Missing signoff name in profile")

    opening_work = _clean_sentence(data["opening_work"], "opening_work")
    role_fit = _clean_sentence(data["role_fit"], "role_fit")
    achievement_1 = _clean_sentence(data["achievement_1"], "achievement_1")
    achievement_2 = _clean_sentence(data["achievement_2"], "achievement_2")
    company_reason = _clean_sentence(data["company_reason"], "company_reason")
    closing = _clean_sentence(data["closing"], "closing")

    return f"""Dear Hiring Manager,

{_job_context_sentence(job)} {opening_work} {role_fit}

{achievement_1} {achievement_2}

{company_reason}

{closing}

{sign_off_name}"""


def _redact_excerpt(raw: str, limit: int = 600) -> str:
    text = str(raw).replace("\r\n", "\n")
    text = re.sub(r"[\w.+-]+@[\w-]+(?:\.[\w-]+)+", "[email]", text)
    text = re.sub(r"\+?\d[\d\s().-]{7,}\d", "[phone]", text)
    if len(text) > limit:
        return text[:limit].rstrip() + "..."
    return text


def _record_failed_attempt(report: dict, attempt: int, raw: str, errors: list[str], kind: str) -> None:
    report.setdefault("failed_attempts", []).append(
        {
            "attempt": attempt,
            "kind": kind,
            "errors": errors,
            "raw_excerpt": _redact_excerpt(raw),
        }
    )


def _matching_job_keywords(job: dict, profile: dict) -> list[str]:
    haystack = f"{job.get('title') or ''}\n{job.get('full_description') or ''}".casefold()
    candidates = (*JOB_KEYWORDS, *_flatten_allowed_skills(profile))
    matches: list[str] = []
    seen: set[str] = set()
    for term in candidates:
        clean = str(term).strip()
        key = clean.casefold()
        if len(key) <= 2:
            found = bool(re.search(rf"\b{re.escape(key)}\b", haystack))
        else:
            found = key in haystack
        if clean and key not in seen and found:
            seen.add(key)
            matches.append(clean)
    return matches[:5]


def _fallback_cover_letter(job: dict, profile: dict) -> str:
    """Build a conservative job-aware cover letter if structured generation fails."""
    personal = profile.get("personal", {})
    sign_off_name = personal.get("preferred_name") or personal.get("full_name", "")
    title = _display_title(job)
    company = _display_company(job)
    location = str(job.get("location") or "").strip()
    keywords = _matching_job_keywords(job, profile)
    job_focus = ", ".join(keywords[:4]) if keywords else "data pipelines, reporting, and data quality"
    location_sentence = f" The location listed is {location}." if location else ""

    return f"""Dear Hiring Manager,

For the {title} role at {company}, the relevant overlap is {job_focus}.{location_sentence} I build Python and SQL data pipelines, reporting models, and validation checks for teams that need clear operational data and dependable releases. The common thread in my recent work has been turning messy operational sources into systems that other teams can trust.

At KPMG, I integrated the Watershed API into ESG reporting workflows using Databricks Asset Bundles, Pydantic models, and Unity Catalog. At Medicine for Business, I deployed openEHR systems with Docker Compose and built Microsoft Fabric and Power BI reporting for hospitals. At Metyis/Adaptfy, I moved IoT workloads to Azure Data Factory, Databricks, and Cosmos DB, then added Great Expectations checks so analysts and stakeholders worked from the same governed datasets.

Your job description points to hands-on data work and ownership across delivery, quality, and reporting. That is close to the systems I have shipped across healthcare, ESG, logistics, and finance. I would bring that same practical bias here: clear pipelines, visible data quality, and enough documentation that the next person can operate the system without guessing.

I would be glad to walk through the tradeoffs, code, and delivery choices behind that work, and to discuss where my experience fits the problems your team wants to solve.

{sign_off_name}"""


def _cover_content_quality(letter: str, mode: str = "normal") -> dict:
    """Check cover-letter length separately from structural validation.

    The validator catches malformed output. This check catches letters that are
    technically complete but too thin to be worth submitting.
    """
    words = len(letter.split())
    errors: list[str] = []
    warnings: list[str] = []

    if mode != "lenient" and words < COVER_MIN_WORDS:
        errors.append(f"Too brief ({words} words). Target {COVER_MIN_WORDS}-{COVER_TARGET_MAX_WORDS} words.")
    if mode == "strict" and words > COVER_TARGET_MAX_WORDS:
        errors.append(f"Too long ({words} words). Max {COVER_TARGET_MAX_WORDS}.")
    elif mode == "normal" and words > COVER_HARD_MAX_WORDS:
        errors.append(f"Too long ({words} words). Max {COVER_HARD_MAX_WORDS}.")
    elif mode == "normal" and words > COVER_TARGET_MAX_WORDS:
        warnings.append(f"Long ({words} words). Target {COVER_TARGET_MAX_WORDS}.")

    return {
        "passed": len(errors) == 0,
        "errors": errors,
        "warnings": warnings,
        "word_count": words,
        "character_count": len(letter),
    }


def _should_use_fallback_early(validation: dict, attempt: int) -> bool:
    """Stop retrying once repeated output is clearly structural junk."""
    if attempt < 1:
        return False
    structural_errors = (
        "Must start with",
        "Must end with",
        "Must include three short body paragraphs",
        "Too short",
        "Looks truncated",
        "Too brief",
    )
    return any(
        any(str(error).startswith(prefix) for prefix in structural_errors) for error in validation.get("errors", [])
    )


def _resume_text_for_job(job: dict, base_resume_text: str) -> str:
    tailored_path = str(job.get("tailored_resume_path") or "").strip()
    if not tailored_path:
        return base_resume_text

    try:
        tailored_text = Path(tailored_path).read_text(encoding="utf-8")
    except OSError as exc:
        log.warning("Could not read tailored resume %s; using base resume: %s", tailored_path, exc)
        return base_resume_text

    return tailored_text if tailored_text.strip() else base_resume_text


# -- Core Generation --------------------------------------------------------


def generate_cover_letter(
    resume_text: str,
    job: dict,
    profile: dict,
    max_retries: int | None = None,
    validation_mode: str = "normal",
) -> tuple[str, dict]:
    """Generate a complete cover letter from structured model output."""
    if max_retries is None:
        max_retries = _cover_max_retries()

    job_text = (
        f"TITLE: {_display_title(job)}\n"
        f"COMPANY: {_display_company(job)}\n"
        f"SOURCE: {job['site']}\n"
        f"LOCATION: {job.get('location', 'N/A')}\n\n"
        f"DESCRIPTION:\n{(job.get('full_description') or '')[:6000]}"
    )

    avoid_notes: list[str] = []
    client = _get_cover_client()
    cl_prompt_base = _build_cover_letter_prompt(profile)
    max_tokens = _cover_max_tokens()
    expected_signoff = profile.get("personal", {}).get("preferred_name") or profile.get("personal", {}).get(
        "full_name",
        "",
    )
    last_validation: dict | None = None
    fallback_reason = "llm_failed_validation"
    report: dict = {
        "attempts": 0,
        "validator": None,
        "status": "pending",
        "validation_mode": validation_mode,
        "generator": "structured_json",
        "fallback": False,
        "fallback_reason": None,
        "cover_model_override": bool(_cover_model_override()),
        "cover_provider_override": _cover_provider_override() or None,
        "model": getattr(client, "model", None),
        "llm_errors": [],
        "failed_attempts": [],
        "content_quality": None,
    }

    for attempt in range(max_retries + 1):
        report["attempts"] = attempt + 1
        prompt = cl_prompt_base
        if avoid_notes:
            prompt += "\n\nFix these issues in the next JSON response:\n" + "\n".join(
                f"- {note}" for note in avoid_notes[-5:]
            )

        messages = [
            {"role": "system", "content": prompt},
            {
                "role": "user",
                "content": (
                    f"TAILORED RESUME CONTEXT:\n{resume_text[:9000]}\n\n---\n\n"
                    f"TARGET JOB:\n{job_text}\n\nReturn the JSON now."
                ),
            },
        ]

        try:
            started = time.time()
            emit_event(
                "cover_letter_attempt_started",
                stage="cover",
                job_title=job.get("title"),
                site=job.get("site"),
                job_url=job.get("url"),
                attempt=attempt + 1,
                max_attempts=max_retries + 1,
                model=report["model"],
                max_tokens=max_tokens,
            )
            log.info(
                "Cover letter attempt %d/%d started | %s | %s",
                attempt + 1,
                max_retries + 1,
                job.get("title", "Untitled"),
                job.get("site", "Unknown"),
            )
            raw = sanitize_text(client.chat(messages, max_tokens=max_tokens, temperature=0.2))
            emit_event(
                "cover_letter_attempt_response",
                stage="cover",
                job_title=job.get("title"),
                site=job.get("site"),
                job_url=job.get("url"),
                attempt=attempt + 1,
                elapsed_seconds=round(time.time() - started, 1),
                raw_chars=len(raw),
            )
        except Exception as exc:
            fallback_reason = "llm_error"
            report["llm_errors"].append({"attempt": attempt + 1, "error": str(exc)})
            log.warning(
                "Cover letter LLM call failed on attempt %d/%d: %s",
                attempt + 1,
                max_retries + 1,
                exc,
            )
            if attempt >= 1:
                break
            continue

        try:
            data = _extract_cover_json(raw)
            letter = _assemble_cover_letter(data, job, profile)
        except ValueError as exc:
            fallback_reason = "llm_bad_json"
            _record_failed_attempt(report, attempt + 1, raw, [str(exc)], "parse_or_structure")
            avoid_notes.append(str(exc))
            if attempt >= 1:
                break
            continue

        validation = validate_cover_letter(letter, mode=validation_mode, expected_signoff=expected_signoff)
        quality = _cover_content_quality(letter, mode=validation_mode)
        last_validation = validation
        report["validator"] = validation
        report["content_quality"] = quality
        if validation["passed"] and quality["passed"]:
            report["status"] = "approved"
            return letter, report

        fallback_reason = "llm_failed_validation"
        errors = [*validation["errors"], *quality["errors"]]
        _record_failed_attempt(report, attempt + 1, raw, errors, "validation")
        avoid_notes.extend(errors)
        log.debug(
            "Cover letter attempt %d/%d failed: %s",
            attempt + 1,
            max_retries + 1,
            errors,
        )
        if _should_use_fallback_early({"errors": errors}, attempt):
            break

    fallback = _fallback_cover_letter(job, profile)
    validation = validate_cover_letter(fallback, mode=validation_mode, expected_signoff=expected_signoff)
    quality = _cover_content_quality(fallback, mode=validation_mode)
    report["fallback"] = True
    report["fallback_reason"] = fallback_reason
    report["llm_failed_validation"] = last_validation
    report["validator"] = validation
    report["content_quality"] = quality
    if validation["passed"] and quality["passed"]:
        report["status"] = "approved"
        return fallback, report

    report["status"] = "failed_validation"
    raise ValueError(
        f"Cover letter validation failed after {max_retries + 1} attempts and fallback also failed: "
        f"{validation}; content_quality={quality}"
    )


def _job_file_prefix(job: dict) -> str:
    """Build a stable, collision-resistant filename prefix for a job."""
    safe_title = re.sub(r"[^\w\s-]", "", job["title"])[:50].strip().replace(" ", "_")
    safe_site = re.sub(r"[^\w\s-]", "", job["site"])[:20].strip().replace(" ", "_")
    digest = sha1(str(job["url"]).encode("utf-8")).hexdigest()[:8]
    return f"{safe_site}_{safe_title}_{digest}"


def _load_jobs_needing_cover_letters(conn, min_score: int = 7, limit: int = 20) -> list:
    """Fetch accepted tailored jobs that still need cover letters."""
    query_limit = max(limit * 5, limit)
    rows = conn.execute(
        "SELECT * FROM jobs "
        "WHERE fit_score >= ? AND tailored_resume_path IS NOT NULL "
        "AND full_description IS NOT NULL "
        "AND (cover_letter_path IS NULL OR cover_letter_path = '') "
        "AND COALESCE(cover_attempts, 0) < ? "
        "AND COALESCE(review_status, '') != 'manual_review' "
        "ORDER BY fit_score DESC LIMIT ?",
        (min_score, MAX_ATTEMPTS, query_limit),
    ).fetchall()

    if not rows:
        return []

    jobs = [dict(zip(row.keys(), row)) for row in rows]
    search_cfg = load_search_config()
    accept_locs, reject_locs = workday._load_location_filter(search_cfg)
    accepted: list[dict] = []
    for job in jobs:
        triage = workday._triage_location(
            job.get("location"),
            accept_locs,
            reject_locs,
            policy="recall_first",
            search_cfg=search_cfg,
        )
        if triage.decision != "accept":
            log.info(
                "Skipping cover letter for %s at %s: location_%s:%s",
                job.get("title"),
                job.get("site"),
                triage.decision,
                triage.reason,
            )
            continue
        accepted.append(job)
        if len(accepted) >= limit:
            break
    return accepted


# -- Batch Entry Point ------------------------------------------------------


def run_cover_letters(min_score: int = 7, limit: int = 20, validation_mode: str = "normal") -> dict:
    """Generate cover letters for high-scoring jobs that have tailored resumes.

    Args:
        min_score:       Minimum fit_score threshold.
        limit:           Maximum jobs to process.
        validation_mode: "strict", "normal", or "lenient".

    Returns:
        {"generated": int, "errors": int, "fallback": int, "elapsed": float}
    """
    profile = load_profile()
    base_resume_text = RESUME_PATH.read_text(encoding="utf-8")
    conn = get_connection()
    jobs = _load_jobs_needing_cover_letters(conn, min_score=min_score, limit=limit)

    if not jobs:
        log.info("No jobs needing cover letters (score >= %d).", min_score)
        return {"generated": 0, "errors": 0, "fallback": 0, "elapsed": 0.0}

    if jobs and not isinstance(jobs[0], dict):
        columns = jobs[0].keys()
        jobs = [dict(zip(columns, row)) for row in jobs]

    COVER_LETTER_DIR.mkdir(parents=True, exist_ok=True)
    log.info(
        "Generating cover letters for %d jobs (score >= %d)...",
        len(jobs),
        min_score,
    )
    t0 = time.time()
    completed = 0
    results: list[dict] = []
    error_count = 0

    for job in jobs:
        completed += 1
        try:
            resume_text = _resume_text_for_job(job, base_resume_text)
            letter, report = generate_cover_letter(resume_text, job, profile, validation_mode=validation_mode)

            prefix = _job_file_prefix(job)

            cl_path = COVER_LETTER_DIR / f"{prefix}_CL.txt"
            cl_path.write_text(letter, encoding="utf-8")
            report_path = COVER_LETTER_DIR / f"{prefix}_CL_REPORT.json"
            report_path.write_text(json.dumps(report, indent=2), encoding="utf-8")

            pdf_path = None
            try:
                from applypilot.scoring.pdf import convert_to_pdf

                pdf_path = str(convert_to_pdf(cl_path, job=job, profile=profile))
            except Exception:
                log.debug("PDF generation failed for %s", cl_path, exc_info=True)

            result = {
                "url": job["url"],
                "path": str(cl_path),
                "pdf_path": pdf_path,
                "title": job["title"],
                "site": job["site"],
                "fallback": bool(report.get("fallback")),
            }
            results.append(result)
            emit_event(
                "cover_letter_finished",
                stage="cover",
                status="ok",
                job_url=job["url"],
                job_title=job["title"],
                site=job["site"],
                cover_letter_path=cl_path,
                report_path=report_path,
                pdf_path=pdf_path,
                fallback=result["fallback"],
            )

            elapsed = time.time() - t0
            rate = completed / elapsed if elapsed > 0 else 0
            status = "FALLBACK" if result["fallback"] else "OK"
            log.info(
                "%d/%d [%s] | %.1f jobs/min | %s",
                completed,
                len(jobs),
                status,
                rate * 60,
                result["title"][:40],
            )
        except Exception as exc:
            result = {
                "url": job["url"],
                "title": job["title"],
                "site": job["site"],
                "path": None,
                "pdf_path": None,
                "fallback": False,
                "error": str(exc),
            }
            error_count += 1
            results.append(result)
            log.error("%d/%d [ERROR] %s -- %s", completed, len(jobs), job["title"][:40], exc)
            emit_error(
                "cover_letter_failed",
                exc,
                stage="cover",
                job_url=job["url"],
                job_title=job["title"],
                site=job["site"],
            )

    now = datetime.now(timezone.utc).isoformat()
    saved = 0
    fallback_count = 0
    for result in results:
        if result.get("path"):
            conn.execute(
                "UPDATE jobs SET cover_letter_path=?, cover_letter_at=?, "
                "cover_attempts=COALESCE(cover_attempts,0)+1 WHERE url=?",
                (result["path"], now, result["url"]),
            )
            saved += 1
            fallback_count += int(bool(result.get("fallback")))
        else:
            conn.execute(
                "UPDATE jobs SET cover_attempts=COALESCE(cover_attempts,0)+1 WHERE url=?",
                (result["url"],),
            )
    conn.commit()

    elapsed = time.time() - t0
    log.info(
        "Cover letters done in %.1fs: %d generated, %d fallback, %d errors",
        elapsed,
        saved,
        fallback_count,
        error_count,
    )

    return {
        "generated": saved,
        "errors": error_count,
        "fallback": fallback_count,
        "elapsed": elapsed,
    }
