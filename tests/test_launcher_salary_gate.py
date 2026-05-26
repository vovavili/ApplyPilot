import json

import applypilot.config as app_config
from applypilot.apply import launcher
from applypilot.database import init_db
from applypilot.scoring import tailor


VALID_COVER = """Dear Hiring Manager,

I built Python and SQL data pipelines for teams that needed reliable reporting instead of fragile spreadsheets. The work used Databricks, Microsoft Fabric, Power BI, and API integrations, which maps well to practical data engineering roles. It also required the less visible parts of the job: clear handoffs, repeatable releases, and validation checks that helped teams trust the numbers before decisions depended on them.

At KPMG, I integrated external API data into ESG reporting workflows and used Databricks Asset Bundles, Pydantic models, and Unity Catalog to make releases easier to review. I also deployed openEHR systems with Docker Compose and built dashboards for several hospitals. At Metyis/Adaptfy, I moved IoT workloads to Azure Data Factory, Databricks, and Cosmos DB, then added Great Expectations checks so analysts and stakeholders worked from the same governed datasets.

Your role points to hands-on delivery across ingestion, modeling, reporting, and quality. That is close to the systems I have shipped across healthcare, ESG, logistics, and finance, usually in settings where the data platform had to become understandable to people outside engineering.

I would be glad to discuss how that mix of data engineering, reporting, and delivery could help your team ship reliable analytics faster.

Test"""


def _write_artifacts(tmp_path, job: dict) -> tuple[str, str]:
    prefix = tailor._job_file_prefix(job)
    resume_path = tmp_path / f"{prefix}.txt"
    cover_path = tmp_path / f"{prefix}_CL.txt"
    resume_path.write_text("Tailored resume text", encoding="utf-8")
    cover_path.write_text(VALID_COVER, encoding="utf-8")
    (tmp_path / f"{prefix}_REPORT.json").write_text(json.dumps({"status": "approved"}), encoding="utf-8")
    (tmp_path / f"{prefix}_CL_REPORT.json").write_text(json.dumps({"status": "approved"}), encoding="utf-8")
    return str(resume_path), str(cover_path)


def _insert_job(conn, tmp_path, *, url: str, salary: str, score: int) -> None:
    job = {"url": url, "title": "Data Engineer", "site": "ExampleCo"}
    resume_path, cover_path = _write_artifacts(tmp_path, job)
    conn.execute(
        """
        INSERT INTO jobs (
            url, title, site, salary, application_url, tailored_resume_path,
            cover_letter_path, fit_score, location, full_description
        )
        VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?, ?)
        """,
        (
            url,
            job["title"],
            job["site"],
            salary,
            url,
            resume_path,
            cover_path,
            score,
            "Remote",
            "Python and data engineering",
        ),
    )
    conn.commit()


def test_acquire_job_skips_lowball_posting_and_claims_next_job(monkeypatch, tmp_path):
    conn = init_db(tmp_path / "applypilot.db")
    _insert_job(conn, tmp_path, url="https://example.com/low", salary="€70,000 - €80,000", score=10)
    _insert_job(conn, tmp_path, url="https://example.com/good", salary="€120,000 - €140,000", score=9)

    monkeypatch.setattr(launcher, "get_connection", lambda: conn)
    monkeypatch.setattr(launcher, "_load_blocked", lambda: (set(), []))
    monkeypatch.setattr(
        launcher.config,
        "load_profile",
        lambda: {"compensation": {"private_minimum": "100000", "salary_currency": "EUR"}},
    )
    monkeypatch.setattr(
        app_config,
        "load_search_config",
        lambda: {
            "location_accept": ["Netherlands", "Remote", "Europe", "EU"],
            "location_reject_non_remote": ["India", "Canada", "United States"],
        },
    )
    monkeypatch.setattr(app_config, "is_manual_ats", lambda _url: False)

    job = launcher.acquire_job(min_score=7, worker_id=2)

    assert job is not None
    assert job["url"] == "https://example.com/good"

    low = conn.execute(
        "SELECT apply_status, apply_error, apply_attempts FROM jobs WHERE url = ?", ("https://example.com/low",)
    ).fetchone()
    good = conn.execute(
        "SELECT apply_status, agent_id FROM jobs WHERE url = ?", ("https://example.com/good",)
    ).fetchone()

    assert low["apply_status"] == "failed"
    assert low["apply_error"] == "not_eligible_salary:posted_salary_below_private_minimum"
    assert low["apply_attempts"] == 99
    assert good["apply_status"] == "in_progress"
    assert good["agent_id"] == "worker-2"

    conn.close()
