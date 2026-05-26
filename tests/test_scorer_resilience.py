import pytest

from applypilot.database import init_db
from applypilot.database import get_jobs_by_stage
from applypilot.database import get_stats
from applypilot import pipeline
from applypilot.scoring import cover_letter
from applypilot.scoring import scorer
from applypilot.scoring import selection


def _insert_score_job(
    conn,
    *,
    url: str,
    title: str,
    strategy: str = "jobspy",
    review_status: str | None = None,
    fit_score: int | None = None,
    tailored_resume_path: str | None = None,
    cover_letter_path: str | None = None,
    location: str = "Amsterdam, Netherlands",
) -> None:
    conn.execute(
        """
        INSERT INTO jobs (
            url, title, site, strategy, discovered_at, full_description, detail_scraped_at,
            review_status, fit_score, tailored_resume_path, cover_letter_path, location
        )
        VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?)
        """,
        (
            url,
            title,
            "ExampleCo",
            strategy,
            "2026-05-25T18:00:00+00:00",
            "Python data engineering job description",
            "2026-05-25T18:00:00+00:00",
            review_status,
            fit_score,
            tailored_resume_path,
            cover_letter_path,
            location,
        ),
    )
    conn.commit()


def test_load_jobs_to_score_skips_manual_review_disabled_workday_and_excluded_titles(monkeypatch, tmp_path):
    conn = init_db(tmp_path / "applypilot.db")
    _insert_score_job(conn, url="https://example.com/jobspy", title="Data Engineer")
    _insert_score_job(
        conn,
        url="https://example.com/manual",
        title="Data Engineer",
        review_status="manual_review",
    )
    _insert_score_job(conn, url="https://example.com/workday", title="Data Engineer", strategy="workday_api")
    _insert_score_job(conn, url="https://example.com/intern", title="Data Engineer Intern")
    monkeypatch.setattr(
        selection.config,
        "load_search_config",
        lambda: {"workday_enabled": False, "exclude_titles": ["intern"]},
    )

    jobs = scorer._load_jobs_to_score(conn)

    assert [job["url"] for job in jobs] == ["https://example.com/jobspy"]
    assert selection.count_jobs_to_score(conn) == 1
    conn.close()


def test_run_scoring_commits_each_job_before_interrupt(monkeypatch, tmp_path):
    conn = init_db(tmp_path / "applypilot.db")
    _insert_score_job(conn, url="https://example.com/first", title="Data Engineer")
    _insert_score_job(conn, url="https://example.com/second", title="Analytics Engineer")
    resume_path = tmp_path / "resume.txt"
    resume_path.write_text("Python data engineer", encoding="utf-8")

    monkeypatch.setattr(scorer, "RESUME_PATH", resume_path)
    monkeypatch.setattr(scorer, "get_connection", lambda: conn)
    monkeypatch.setattr(selection.config, "load_search_config", lambda: {"workday_enabled": False})
    monkeypatch.setattr(
        scorer,
        "_load_jobs_to_score",
        lambda *_args, **_kwargs: [
            {"url": "https://example.com/first", "title": "Data Engineer", "site": "ExampleCo"},
            {"url": "https://example.com/second", "title": "Analytics Engineer", "site": "ExampleCo"},
        ],
    )

    def fake_score(_resume_text, job):
        if job["url"].endswith("/second"):
            raise KeyboardInterrupt
        return {"score": 8, "keywords": "Python, SQL", "reasoning": "Strong match."}

    monkeypatch.setattr(scorer, "score_job", fake_score)

    with pytest.raises(KeyboardInterrupt):
        scorer.run_scoring()

    first = conn.execute(
        "SELECT fit_score, score_reasoning FROM jobs WHERE url = ?", ("https://example.com/first",)
    ).fetchone()
    second = conn.execute("SELECT fit_score FROM jobs WHERE url = ?", ("https://example.com/second",)).fetchone()

    assert first["fit_score"] == 8
    assert "Python, SQL" in first["score_reasoning"]
    assert second["fit_score"] is None
    conn.close()


def test_pipeline_pending_score_uses_same_filter_as_scorer(monkeypatch, tmp_path):
    conn = init_db(tmp_path / "applypilot.db")
    _insert_score_job(conn, url="https://example.com/jobspy", title="Data Engineer")
    _insert_score_job(
        conn,
        url="https://example.com/manual",
        title="Data Engineer",
        review_status="manual_review",
    )
    _insert_score_job(conn, url="https://example.com/workday", title="Data Engineer", strategy="workday_api")
    _insert_score_job(conn, url="https://example.com/intern", title="Data Engineer Intern")

    monkeypatch.setattr(pipeline, "get_connection", lambda: conn)
    monkeypatch.setattr(
        selection.config,
        "load_search_config",
        lambda: {"workday_enabled": False, "exclude_titles": ["intern"]},
    )

    assert pipeline._count_pending("score") == 1
    conn.close()


def test_manual_review_jobs_are_not_tailor_or_cover_candidates(monkeypatch, tmp_path):
    conn = init_db(tmp_path / "applypilot.db")
    accepted_resume = tmp_path / "accepted_resume.txt"
    manual_resume = tmp_path / "manual_resume.txt"
    accepted_resume.write_text("accepted", encoding="utf-8")
    manual_resume.write_text("manual", encoding="utf-8")
    _insert_score_job(
        conn,
        url="https://example.com/accepted",
        title="Data Engineer",
        fit_score=8,
    )
    _insert_score_job(
        conn,
        url="https://example.com/manual",
        title="Data Engineer",
        review_status="manual_review",
        fit_score=8,
    )

    monkeypatch.setattr(pipeline, "get_connection", lambda: conn)

    tailor_jobs = get_jobs_by_stage(conn=conn, stage="pending_tailor", min_score=7)
    assert [job["url"] for job in tailor_jobs] == ["https://example.com/accepted"]
    assert get_stats(conn)["untailored_eligible"] == 1
    assert pipeline._count_pending("tailor") == 1

    conn.execute(
        "UPDATE jobs SET tailored_resume_path = CASE url "
        "WHEN 'https://example.com/accepted' THEN ? "
        "WHEN 'https://example.com/manual' THEN ? END",
        (str(accepted_resume), str(manual_resume)),
    )
    conn.commit()

    cover_jobs = cover_letter._load_jobs_needing_cover_letters(conn, min_score=7)
    assert [job["url"] for job in cover_jobs] == ["https://example.com/accepted"]
    assert pipeline._count_pending("cover") == 1
    conn.close()


def test_cover_selection_excludes_foreign_locations(tmp_path, monkeypatch):
    conn = init_db(tmp_path / "applypilot.db")
    accepted_resume = tmp_path / "accepted.txt"
    foreign_resume = tmp_path / "foreign.txt"
    accepted_resume.write_text("accepted", encoding="utf-8")
    foreign_resume.write_text("foreign", encoding="utf-8")
    _insert_score_job(
        conn,
        url="https://example.com/accepted",
        title="Data Engineer",
        fit_score=8,
        tailored_resume_path=str(accepted_resume),
        location="Amsterdam, Netherlands",
    )
    _insert_score_job(
        conn,
        url="https://example.com/foreign",
        title="SW Engineer",
        fit_score=8,
        tailored_resume_path=str(foreign_resume),
        location="Buenos Aires, Argentina",
    )
    monkeypatch.setattr(pipeline, "get_connection", lambda: conn)

    cover_jobs = cover_letter._load_jobs_needing_cover_letters(conn, min_score=7)

    assert [job["url"] for job in cover_jobs] == ["https://example.com/accepted"]
    assert pipeline._count_pending("cover") == 1
    conn.close()
