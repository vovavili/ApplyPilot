import json
from pathlib import Path

import pytest
from hypothesis import assume, given
from hypothesis import strategies as st

import applypilot.config as app_config
from applypilot.apply import launcher
from applypilot.database import get_jobs_by_stage, get_stats, init_db
from applypilot.readiness import (
    audit_jobs,
    check_job_ready,
    count_ready_to_apply,
    expected_url_digest,
    fix_not_ready_jobs,
)
from applypilot.scoring import cover_letter, tailor


VALID_COVER = """Dear Hiring Manager,

I built Python and SQL data pipelines for clinical and financial reporting teams that needed cleaner operational data and fewer manual checks. The work combined Databricks, Microsoft Fabric, Power BI, and API integrations, which matches the kind of practical data engineering this role needs. It also required plain ownership: repeatable releases, clear data contracts, and validation that made problems visible before reports reached stakeholders.

At KPMG, I integrated external API data into ESG reporting workflows and used Databricks Asset Bundles, Pydantic models, and Unity Catalog to make releases easier to review. I have also deployed openEHR systems with Docker Compose and built dashboards for several hospitals. At Metyis/Adaptfy, I moved IoT workloads to Azure Data Factory, Databricks, and Cosmos DB, then added Great Expectations checks so analysts and stakeholders worked from the same governed datasets.

Your role points to hands-on delivery across ingestion, modeling, reporting, and quality. That is close to the systems I have shipped across healthcare, ESG, logistics, and finance, usually in settings where the data platform had to become understandable to people outside engineering.

I would be glad to discuss how that mix of data engineering, reporting, and pragmatic delivery could help your team ship reliable analytics faster.

Test"""


def _profile() -> dict:
    return {
        "personal": {"full_name": "Test Candidate", "preferred_name": "Test"},
        "compensation": {"private_minimum": "100000", "salary_currency": "EUR"},
    }


def _search_config() -> dict:
    return {
        "location_accept": ["Netherlands", "Remote", "Europe", "EU"],
        "location_reject_non_remote": ["India", "Canada", "United States", "Argentina", "Costa Rica", "Spain"],
    }


def _train_search_config() -> dict:
    return {
        "location_accept": ["Remote", "Europe", "EU", "EMEA"],
        "location_reject_non_remote": ["India", "Canada", "United States", "Maastricht", "Limburg"],
        "location_train_policy": {
            "enabled": True,
            "max_minutes": 100,
            "unknown_city": "manual_review",
            "anchors": [
                {"station": "Den Haag Centraal", "code": "GVC"},
                {"station": "Rotterdam Centraal", "code": "RTD"},
            ],
            "source": {
                "static_table": True,
                "ns_api_fallback": False,
                "cache_path": "",
                "max_api_lookups_per_run": 0,
            },
        },
    }


def _job(url: str = "https://example.com/jobs/1", **overrides) -> dict:
    job = {
        "url": url,
        "title": "Data Engineer",
        "site": "ExampleCo",
        "location": "Amsterdam, Netherlands",
        "application_url": url,
        "fit_score": 8,
        "full_description": "Python data engineering role.",
    }
    job.update(overrides)
    return job


def _write_artifacts(
    tmp_path: Path, job: dict, *, tailor_status: str = "approved", cover_text: str = VALID_COVER
) -> tuple[str, str]:
    prefix = tailor._job_file_prefix(job)
    resume_path = tmp_path / f"{prefix}.txt"
    cover_path = tmp_path / f"{prefix}_CL.txt"
    resume_path.write_text("Tailored resume text", encoding="utf-8")
    cover_path.write_text(cover_text, encoding="utf-8")
    (tmp_path / f"{prefix}_REPORT.json").write_text(json.dumps({"status": tailor_status}), encoding="utf-8")
    (tmp_path / f"{prefix}_CL_REPORT.json").write_text(json.dumps({"status": "approved"}), encoding="utf-8")
    return str(resume_path), str(cover_path)


def _insert_prepped_job(conn, job: dict, resume_path: str | None, cover_path: str | None) -> None:
    conn.execute(
        """
        INSERT INTO jobs (
            url, title, site, location, application_url, discovered_at,
            full_description, detail_scraped_at, fit_score,
            tailored_resume_path, cover_letter_path
        )
        VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?)
        """,
        (
            job["url"],
            job["title"],
            job["site"],
            job["location"],
            job["application_url"],
            "2026-05-25T18:00:00+00:00",
            job["full_description"],
            "2026-05-25T18:00:00+00:00",
            job["fit_score"],
            resume_path,
            cover_path,
        ),
    )
    conn.commit()


def _patch_environment(monkeypatch, conn) -> None:
    monkeypatch.setattr(app_config, "load_search_config", _search_config)
    monkeypatch.setattr(app_config, "load_profile", _profile)
    monkeypatch.setattr(app_config, "is_manual_ats", lambda _url: False)
    monkeypatch.setattr(launcher, "get_connection", lambda: conn)
    monkeypatch.setattr(launcher, "_load_blocked", lambda: (set(), []))


def test_ready_job_passes_stats_stage_and_apply_acquire(monkeypatch, tmp_path):
    conn = init_db(tmp_path / "applypilot.db")
    job = _job()
    resume_path, cover_path = _write_artifacts(tmp_path, job)
    _insert_prepped_job(conn, job, resume_path, cover_path)
    _patch_environment(monkeypatch, conn)

    row = conn.execute("SELECT * FROM jobs WHERE url = ?", (job["url"],)).fetchone()
    result = check_job_ready(row, conn, profile=_profile())
    acquired = launcher.acquire_job(min_score=7, worker_id=3)

    assert result.ready
    assert count_ready_to_apply(conn) == 1
    assert get_stats(conn)["ready_to_apply"] == 1
    assert [j["url"] for j in get_jobs_by_stage(conn=conn, stage="pending_apply", min_score=7)] == [job["url"]]
    assert acquired["url"] == job["url"]
    assert tuple(conn.execute("SELECT apply_status, agent_id FROM jobs").fetchone()) == ("in_progress", "worker-3")
    conn.close()


def test_readiness_blocks_foreign_location_and_judge_warning(monkeypatch, tmp_path):
    conn = init_db(tmp_path / "applypilot.db")
    foreign = _job("https://example.com/jobs/foreign", location="Bangalore, India")
    warning = _job("https://example.com/jobs/warning")
    _insert_prepped_job(conn, foreign, *_write_artifacts(tmp_path, foreign))
    _insert_prepped_job(
        conn, warning, *_write_artifacts(tmp_path, warning, tailor_status="approved_with_judge_warning")
    )
    _patch_environment(monkeypatch, conn)

    audited = audit_jobs(conn)
    reasons = {item["url"]: item["readiness"]["reasons"] for item in audited}

    assert any(reason.startswith("location_reject") for reason in reasons[foreign["url"]])
    assert "tailored_resume_report_status:approved_with_judge_warning" in reasons[warning["url"]]
    assert count_ready_to_apply(conn) == 0
    conn.close()


@pytest.mark.parametrize(
    "location",
    [
        "Seattle, WA",
        "Toronto",
        "Shanghai",
        "Zug, Switzerland",
        "Bangalore, India",
        "Madrid HQ (KES51610)",
        "Remote-Canada-British Columbia",
        "Buenos Aires, Argentina",
        "Heredia, Costa Rica",
    ],
)
def test_readiness_blocks_current_bad_location_examples(monkeypatch, tmp_path, location):
    conn = init_db(tmp_path / "applypilot.db")
    job = _job(f"https://example.com/jobs/{location.replace(' ', '-')}", location=location)
    _insert_prepped_job(conn, job, *_write_artifacts(tmp_path, job))
    _patch_environment(monkeypatch, conn)

    row = conn.execute("SELECT * FROM jobs WHERE url = ?", (job["url"],)).fetchone()
    readiness = check_job_ready(row, conn, profile=_profile())

    assert not readiness.ready
    assert any(reason.startswith("location_") for reason in readiness.reasons)
    assert count_ready_to_apply(conn) == 0
    conn.close()


@pytest.mark.parametrize(
    ("location", "reason_part"),
    [
        ("Maastricht, Limburg, Netherlands", "location_reject:rejected_non_remote_foreign"),
        ("North Brabant, Netherlands", "location_manual_review:province_only_location"),
        ("Enkhuizen, North Holland, Netherlands", "location_manual_review:unknown_city"),
        ("Zaltbommel, Gelderland, Netherlands", "location_manual_review:unknown_city"),
        ("Renswoude, Utrecht, Netherlands", "location_manual_review:unknown_city"),
    ],
)
def test_train_commute_policy_blocks_existing_bad_dutch_rows(monkeypatch, tmp_path, location, reason_part):
    conn = init_db(tmp_path / "applypilot.db")
    job = _job(f"https://example.com/jobs/{location.replace(' ', '-')}", location=location)
    _insert_prepped_job(conn, job, *_write_artifacts(tmp_path, job))
    _patch_environment(monkeypatch, conn)
    monkeypatch.setattr(app_config, "load_search_config", _train_search_config)

    row = conn.execute("SELECT * FROM jobs WHERE url = ?", (job["url"],)).fetchone()
    readiness = check_job_ready(row, conn, profile=_profile())

    assert not readiness.ready
    assert reason_part in readiness.reasons
    assert count_ready_to_apply(conn) == 0
    assert launcher.acquire_job(target_url=job["url"], min_score=7, worker_id=1) is None
    conn.close()


def test_train_commute_policy_keeps_randstad_rows_ready(monkeypatch, tmp_path):
    conn = init_db(tmp_path / "applypilot.db")
    job = _job("https://example.com/jobs/randstad", location="Breda, North Brabant, Netherlands")
    _insert_prepped_job(conn, job, *_write_artifacts(tmp_path, job))
    _patch_environment(monkeypatch, conn)
    monkeypatch.setattr(app_config, "load_search_config", _train_search_config)

    row = conn.execute("SELECT * FROM jobs WHERE url = ?", (job["url"],)).fetchone()
    readiness = check_job_ready(row, conn, profile=_profile())

    assert readiness.ready
    assert count_ready_to_apply(conn) == 1
    assert launcher.acquire_job(target_url=job["url"], min_score=7, worker_id=1)["url"] == job["url"]
    conn.close()


def test_readiness_blocks_duplicate_artifact_paths(monkeypatch, tmp_path):
    conn = init_db(tmp_path / "applypilot.db")
    first = _job("https://example.com/jobs/1")
    second = _job("https://example.com/jobs/2")
    resume_path, cover_path = _write_artifacts(tmp_path, first)
    _insert_prepped_job(conn, first, resume_path, cover_path)
    _insert_prepped_job(conn, second, resume_path, cover_path)
    _patch_environment(monkeypatch, conn)

    audited = audit_jobs(conn)

    assert count_ready_to_apply(conn) == 0
    assert all(
        {"tailored_resume_duplicate_path", "cover_letter_duplicate_path"}.issubset(item["readiness"]["reasons"])
        for item in audited
    )
    conn.close()


def test_audit_fix_marks_unsafe_rows_and_clears_bad_artifacts(monkeypatch, tmp_path):
    conn = init_db(tmp_path / "applypilot.db")
    foreign = _job("https://example.com/jobs/foreign", location="2 Locations")
    old_path = _job("https://example.com/jobs/old")
    resume_path, cover_path = _write_artifacts(tmp_path, foreign)
    old_resume = tmp_path / "old_resume.txt"
    old_cover = tmp_path / "old_cover_CL.txt"
    old_resume.write_text("old resume", encoding="utf-8")
    old_cover.write_text(VALID_COVER, encoding="utf-8")
    _insert_prepped_job(conn, foreign, resume_path, cover_path)
    _insert_prepped_job(conn, old_path, str(old_resume), str(old_cover))
    _patch_environment(monkeypatch, conn)

    counts = fix_not_ready_jobs(conn, audit_jobs(conn))
    rows = {row["url"]: row for row in conn.execute("SELECT * FROM jobs").fetchall()}

    assert counts["manual_review"] == 1
    assert rows[foreign["url"]]["review_status"] == "manual_review"
    assert rows[old_path["url"]]["tailored_resume_path"] is None
    assert rows[old_path["url"]]["cover_letter_path"] is None
    conn.close()


def test_acquire_job_refuses_target_that_fails_readiness(monkeypatch, tmp_path):
    conn = init_db(tmp_path / "applypilot.db")
    job = _job()
    resume_path, cover_path = _write_artifacts(
        tmp_path,
        job,
        cover_text="Dear Hiring Manager,\n\nI built Python data pipelines. This",
    )
    _insert_prepped_job(conn, job, resume_path, cover_path)
    _patch_environment(monkeypatch, conn)

    acquired = launcher.acquire_job(target_url=job["url"], min_score=7, worker_id=1)

    assert acquired is None
    assert conn.execute("SELECT apply_status FROM jobs").fetchone()[0] is None
    conn.close()


@pytest.mark.parametrize("application_url", ["", "None", " none ", "NULL", "nan", "N/A", "ftp://example.com/apply"])
def test_readiness_allows_placeholder_application_url_when_job_url_is_usable(monkeypatch, tmp_path, application_url):
    conn = init_db(tmp_path / "applypilot.db")
    job = _job(application_url=application_url)
    _insert_prepped_job(conn, job, *_write_artifacts(tmp_path, job))
    _patch_environment(monkeypatch, conn)

    row = conn.execute("SELECT * FROM jobs WHERE url = ?", (job["url"],)).fetchone()
    readiness = check_job_ready(row, conn, profile=_profile())

    assert readiness.ready
    assert "missing_application_url" not in readiness.reasons
    assert launcher.acquire_job(target_url=job["url"], min_score=7, worker_id=1)["url"] == job["url"]
    conn.close()


def test_readiness_blocks_when_both_application_and_job_urls_are_unusable(monkeypatch, tmp_path):
    conn = init_db(tmp_path / "applypilot.db")
    job = _job(url="None", application_url="None")
    _insert_prepped_job(conn, job, *_write_artifacts(tmp_path, {"url": "https://example.com/jobs/1", **job}))
    _patch_environment(monkeypatch, conn)

    row = conn.execute("SELECT * FROM jobs").fetchone()
    readiness = check_job_ready(row, conn, profile=_profile())

    assert not readiness.ready
    assert "missing_application_url" in readiness.reasons
    assert launcher.acquire_job(target_url="None", min_score=7, worker_id=1) is None
    conn.close()


_TEXT = st.text(
    alphabet=st.characters(blacklist_categories=("Cs",), blacklist_characters="\x00"),
    min_size=1,
    max_size=80,
)


@given(url=_TEXT, title=_TEXT, site=_TEXT)
def test_job_file_prefix_is_stable_safe_and_shared(url, title, site):
    job = {"url": url, "title": title, "site": site}

    prefix = tailor._job_file_prefix(job)

    assert prefix == tailor._job_file_prefix(job)
    assert prefix == cover_letter._job_file_prefix(job)
    assert expected_url_digest(url) in prefix
    assert not any(char in prefix for char in '<>:"/\\|?*\x00')


@given(url_a=_TEXT, url_b=_TEXT, title=_TEXT, site=_TEXT)
def test_job_file_prefix_changes_when_url_digest_changes(url_a, url_b, title, site):
    assume(expected_url_digest(url_a) != expected_url_digest(url_b))

    first = tailor._job_file_prefix({"url": url_a, "title": title, "site": site})
    second = tailor._job_file_prefix({"url": url_b, "title": title, "site": site})

    assert first != second
