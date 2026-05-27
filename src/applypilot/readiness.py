"""Strict readiness checks for the auto-apply queue."""

from __future__ import annotations

import json
import sqlite3
from dataclasses import dataclass, field
from datetime import datetime
from hashlib import sha1
from pathlib import Path
from typing import Any

from applypilot import config
from applypilot.discovery import workday
from applypilot.events import emit_event
from applypilot.scoring.validator import validate_cover_letter

READY_STATUS_APPROVED = "approved"
MISSING_URL_VALUES = {"", "none", "null", "nan", "n/a", "na"}


@dataclass
class ReadinessResult:
    url: str
    title: str | None
    site: str | None
    ready: bool = True
    reasons: list[str] = field(default_factory=list)
    warnings: list[str] = field(default_factory=list)

    def fail(self, reason: str) -> None:
        self.ready = False
        self.reasons.append(reason)

    def as_dict(self) -> dict[str, Any]:
        return {
            "url": self.url,
            "title": self.title,
            "site": self.site,
            "ready": self.ready,
            "reasons": self.reasons,
            "warnings": self.warnings,
        }


def expected_url_digest(url: str) -> str:
    """Return the short URL digest used in generated artifact filenames."""
    return sha1(str(url).encode("utf-8")).hexdigest()[:8]


def usable_url(value: Any) -> str | None:
    """Return a usable HTTP(S) URL, treating common scraped placeholders as missing."""
    text = str(value or "").strip()
    if text.casefold() in MISSING_URL_VALUES:
        return None
    if not text.startswith(("http://", "https://")):
        return None
    return text


def check_job_ready(
    job_row: sqlite3.Row | dict,
    conn: sqlite3.Connection,
    min_score: int = 7,
    profile: dict | None = None,
    emit: bool = False,
) -> ReadinessResult:
    """Validate that a job can safely enter the apply queue."""
    job = _row_to_dict(job_row)
    url = str(job.get("url") or "")
    result = ReadinessResult(url=url, title=job.get("title"), site=job.get("site"))

    _check_required_fields(job, result, min_score)
    _check_location(job, result)
    _check_artifact("tailored_resume_path", job, conn, result, report_suffix="_REPORT.json")
    _check_artifact("cover_letter_path", job, conn, result, report_suffix="_CL_REPORT.json")
    _check_tailor_report(job, result)
    _check_cover_letter(job, result, profile=profile)

    if emit and not result.ready:
        emit_event(
            "readiness_failed",
            level="warning",
            stage="apply",
            status="blocked",
            job_url=url,
            job_title=job.get("title"),
            site=job.get("site"),
            reasons=result.reasons,
        )
    return result


def load_ready_jobs(conn: sqlite3.Connection, min_score: int = 7, limit: int = 0) -> list[dict]:
    """Return jobs that pass the strict readiness checker."""
    rows = _artifact_candidate_rows(conn, min_score=min_score)
    ready: list[dict] = []
    profile = _load_profile_or_none()
    for row in rows:
        if check_job_ready(row, conn, min_score=min_score, profile=profile).ready:
            ready.append(_row_to_dict(row))
            if limit > 0 and len(ready) >= limit:
                break
    return ready


def count_ready_to_apply(conn: sqlite3.Connection, min_score: int = 7) -> int:
    """Count jobs that pass the same readiness checks used by apply."""
    return len(load_ready_jobs(conn, min_score=min_score))


def audit_jobs(conn: sqlite3.Connection, min_score: int = 7) -> list[dict]:
    """Return readiness audit rows for application-prep candidates."""
    profile = _load_profile_or_none()
    audited = []
    for row in _artifact_candidate_rows(conn, min_score=min_score):
        job = _row_to_dict(row)
        result = check_job_ready(job, conn, min_score=min_score, profile=profile)
        audited.append({**job, "readiness": result.as_dict()})
    return audited


def fix_not_ready_jobs(conn: sqlite3.Connection, audited: list[dict]) -> dict[str, int]:
    """Clear invalid artifact paths and mark unsafe rows for manual review."""
    counts = {"manual_review": 0, "cleared_tailor": 0, "cleared_cover": 0}

    for item in audited:
        readiness = item["readiness"]
        if readiness["ready"]:
            continue
        url = item["url"]
        reasons = set(readiness["reasons"])

        if _should_mark_manual_review(reasons):
            conn.execute(
                "UPDATE jobs SET review_status='manual_review', review_reason=? WHERE url=?",
                ("readiness_audit:" + ",".join(sorted(reasons))[:400], url),
            )
            counts["manual_review"] += 1

        if _should_clear_tailor(reasons):
            conn.execute(
                "UPDATE jobs SET tailored_resume_path=NULL, tailored_at=NULL, tailor_attempts=0, "
                "cover_letter_path=NULL, cover_letter_at=NULL, cover_attempts=0 WHERE url=?",
                (url,),
            )
            counts["cleared_tailor"] += 1
            counts["cleared_cover"] += 1
        elif _should_clear_cover(reasons):
            conn.execute(
                "UPDATE jobs SET cover_letter_path=NULL, cover_letter_at=NULL, cover_attempts=0 WHERE url=?",
                (url,),
            )
            counts["cleared_cover"] += 1

    conn.commit()
    return counts


def backup_database(conn: sqlite3.Connection) -> Path:
    """Create a timestamped SQLite backup before audit fixes."""
    config.ensure_dirs()
    stamp = datetime.now().strftime("%Y%m%d-%H%M%S")
    backup_path = config.APP_DIR / f"applypilot.{stamp}.before-audit-fix.db"
    with sqlite3.connect(backup_path) as backup:
        conn.backup(backup)
    return backup_path


def _artifact_candidate_rows(conn: sqlite3.Connection, min_score: int) -> list[sqlite3.Row]:
    return conn.execute(
        """
        SELECT * FROM jobs
        WHERE applied_at IS NULL
          AND fit_score >= ?
          AND (tailored_resume_path IS NOT NULL OR cover_letter_path IS NOT NULL)
        ORDER BY fit_score DESC, site, title, url
        """,
        (min_score,),
    ).fetchall()


def _row_to_dict(row: sqlite3.Row | dict) -> dict:
    if isinstance(row, dict):
        return row
    return dict(zip(row.keys(), row))


def _check_required_fields(job: dict, result: ReadinessResult, min_score: int) -> None:
    if job.get("review_status") == "manual_review":
        result.fail("manual_review")
    if (job.get("fit_score") or 0) < min_score:
        result.fail("score_below_minimum")
    if not usable_url(job.get("application_url")) and not usable_url(job.get("url")):
        result.fail("missing_application_url")
    if not job.get("tailored_resume_path"):
        result.fail("missing_tailored_resume_path")
    if not job.get("cover_letter_path"):
        result.fail("missing_cover_letter_path")


def _check_location(job: dict, result: ReadinessResult) -> None:
    search_cfg = config.load_search_config()
    accept_locs, reject_locs = workday._load_location_filter(search_cfg)
    triage = workday._triage_location(
        job.get("location"),
        accept_locs,
        reject_locs,
        policy="recall_first",
        search_cfg=search_cfg,
    )
    if triage.decision != "accept":
        result.fail(f"location_{triage.decision}:{triage.reason}")


def _check_artifact(
    field_name: str,
    job: dict,
    conn: sqlite3.Connection,
    result: ReadinessResult,
    *,
    report_suffix: str,
) -> None:
    path_text = job.get(field_name)
    if not path_text:
        return

    path = Path(path_text)
    reason_prefix = field_name.replace("_path", "")
    if not path.exists():
        result.fail(f"{reason_prefix}_missing_file")
        return
    if path.stat().st_size <= 0:
        result.fail(f"{reason_prefix}_empty_file")

    digest = expected_url_digest(job.get("url", ""))
    if digest not in path.stem:
        result.fail(f"{reason_prefix}_missing_url_hash")

    duplicate_count = conn.execute(
        f"SELECT COUNT(*) FROM jobs WHERE url != ? AND applied_at IS NULL AND {field_name} = ?",
        (job.get("url"), str(path)),
    ).fetchone()[0]
    if duplicate_count:
        result.fail(f"{reason_prefix}_duplicate_path")

    report_path = _report_path(path, report_suffix=report_suffix)
    if not report_path.exists():
        result.fail(f"{reason_prefix}_missing_report")


def _check_tailor_report(job: dict, result: ReadinessResult) -> None:
    path_text = job.get("tailored_resume_path")
    if not path_text:
        return

    report = _read_json(_report_path(Path(path_text), report_suffix="_REPORT.json"))
    if report is None:
        return

    status = report.get("status")
    if status != READY_STATUS_APPROVED:
        result.fail(f"tailored_resume_report_status:{status or 'missing'}")


def _check_cover_letter(job: dict, result: ReadinessResult, profile: dict | None) -> None:
    path_text = job.get("cover_letter_path")
    if not path_text:
        return

    report = _read_json(_report_path(Path(path_text), report_suffix="_CL_REPORT.json"))
    if report and report.get("status") != READY_STATUS_APPROVED:
        result.fail(f"cover_letter_report_status:{report.get('status') or 'missing'}")

    try:
        letter = Path(path_text).read_text(encoding="utf-8")
    except OSError:
        return

    expected_signoff = None
    if profile:
        personal = profile.get("personal", {})
        expected_signoff = personal.get("preferred_name") or personal.get("full_name")

    validation = validate_cover_letter(letter, expected_signoff=expected_signoff)
    for error in validation["errors"]:
        result.fail(f"cover_letter_validation:{error}")
    result.warnings.extend(f"cover_letter_validation:{warning}" for warning in validation["warnings"])


def _report_path(path: Path, *, report_suffix: str) -> Path:
    if report_suffix == "_REPORT.json":
        return path.with_name(f"{path.stem}{report_suffix}")
    if path.stem.endswith("_CL"):
        return path.with_name(f"{path.stem}_REPORT.json")
    return path.with_name(f"{path.stem}{report_suffix}")


def _read_json(path: Path) -> dict | None:
    if not path.exists():
        return None
    try:
        return json.loads(path.read_text(encoding="utf-8"))
    except (OSError, json.JSONDecodeError):
        return None


def _load_profile_or_none() -> dict | None:
    try:
        return config.load_profile()
    except FileNotFoundError:
        return None


def _should_mark_manual_review(reasons: set[str]) -> bool:
    return any(reason.startswith("location_") for reason in reasons) or any(
        reason.startswith("tailored_resume_report_status:approved_with_judge_warning") for reason in reasons
    )


def _should_clear_tailor(reasons: set[str]) -> bool:
    return any(
        any(reason.startswith(prefix) for reason in reasons)
        for prefix in (
            "tailored_resume_missing_file",
            "tailored_resume_empty_file",
            "tailored_resume_missing_url_hash",
            "tailored_resume_duplicate_path",
            "tailored_resume_missing_report",
            "tailored_resume_report_status:",
        )
    )


def _should_clear_cover(reasons: set[str]) -> bool:
    return any(
        any(reason.startswith(prefix) for reason in reasons)
        for prefix in (
            "cover_letter_missing_file",
            "cover_letter_empty_file",
            "cover_letter_missing_url_hash",
            "cover_letter_duplicate_path",
            "cover_letter_missing_report",
            "cover_letter_report_status:",
            "cover_letter_validation:",
        )
    )
