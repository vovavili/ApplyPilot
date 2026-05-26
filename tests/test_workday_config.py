import logging
from urllib.error import HTTPError

from applypilot.database import get_stats, init_db
from applypilot.discovery import workday
from applypilot.scoring.selection import count_jobs_to_score

import pytest


def _employer(name="ExampleCo"):
    return {
        "name": name,
        "tenant": "example",
        "site_id": "External",
        "base_url": "https://example.wd1.myworkdayjobs.com",
    }


def test_workday_discovery_can_be_disabled_from_search_config(monkeypatch):
    monkeypatch.setattr(workday.config, "load_search_config", lambda: {"workday_enabled": False})
    monkeypatch.setattr(
        workday,
        "load_employers",
        lambda: (_ for _ in ()).throw(AssertionError("disabled Workday should not load employers")),
    )

    result = workday.run_workday_discovery()

    assert result == {"found": 0, "new": 0, "existing": 0, "queries": 0, "skipped": True}


def test_workday_discovery_skips_empty_allowlist_without_broad_crawl(monkeypatch):
    monkeypatch.setattr(
        workday.config,
        "load_search_config",
        lambda: {
            "workday_enabled": True,
            "workday_allow_broad_crawl": False,
            "workday_employers": [],
            "queries": [{"query": "Data Engineer", "tier": 1}],
        },
    )

    result = workday.run_workday_discovery(employers={"example": _employer()})

    assert result == {"found": 0, "new": 0, "existing": 0, "queries": 0, "skipped": True}


def test_load_employers_merges_user_registry_and_user_wins(monkeypatch, tmp_path):
    package_dir = tmp_path / "package"
    app_dir = tmp_path / "app"
    package_dir.mkdir()
    app_dir.mkdir()
    (package_dir / "employers.yaml").write_text(
        """
employers:
  shared:
    name: "Package Shared"
    tenant: "package"
    site_id: "External"
    base_url: "https://package.wd1.myworkdayjobs.com"
  packaged_only:
    name: "Packaged Only"
    tenant: "packaged"
    site_id: "External"
    base_url: "https://packaged.wd1.myworkdayjobs.com"
""",
        encoding="utf-8",
    )
    (app_dir / "workday_employers.yaml").write_text(
        """
employers:
  shared:
    name: "User Shared"
    tenant: "user"
    site_id: "External"
    base_url: "https://user.wd1.myworkdayjobs.com"
  user_only:
    name: "User Only"
    tenant: "useronly"
    site_id: "External"
    base_url: "https://useronly.wd1.myworkdayjobs.com"
""",
        encoding="utf-8",
    )
    monkeypatch.setattr(workday, "CONFIG_DIR", package_dir)
    monkeypatch.setattr(workday.config, "APP_DIR", app_dir)

    employers = workday.load_employers()

    assert set(employers) == {"shared", "packaged_only", "user_only"}
    assert employers["shared"]["name"] == "User Shared"


def test_validate_employers_warns_and_skips_malformed(caplog):
    caplog.set_level(logging.WARNING)

    valid = workday._validate_employers(
        {
            "good": _employer(),
            "bad": {"name": "Broken"},
            "not_mapping": "lol nope",
        },
        source="test",
    )

    assert list(valid) == ["good"]
    assert "Skipping malformed Workday employer bad" in caplog.text
    assert "Skipping malformed Workday employer not_mapping" in caplog.text


def test_workday_location_filter_rejects_blank_locations():
    assert not workday._location_ok("", ["Netherlands"], [])
    assert not workday._location_ok(None, ["Netherlands"], [])
    assert workday._location_ok("Amsterdam, North Holland, Netherlands", ["Netherlands"], [])
    assert workday._location_ok("Remote", ["Netherlands"], [])


def test_workday_location_triage_recall_first():
    accept = ["Netherlands", "Remote", "Europe", "European Union", "EU"]
    reject = ["United States", "Canada", "India"]

    assert workday._triage_location("Amsterdam, Netherlands", accept, reject).reason == "accepted_nl_or_configured"
    assert workday._triage_location("Remote - EMEA", accept, reject).reason == "accepted_europe_emea"
    assert workday._triage_location("Europe", accept, reject).reason == "accepted_europe_emea"
    assert workday._triage_location("", accept, reject).decision == "manual_review"
    assert workday._triage_location("Multiple Locations", accept, reject).decision == "manual_review"
    assert workday._triage_location("Toronto, Canada", accept, reject).decision == "reject"
    assert workday._triage_location("Remote-Canada-British Columbia", accept, reject).decision == "reject"


@pytest.mark.parametrize(
    "location",
    [
        "Amsterdam, Netherlands",
        "Remote - EMEA",
        "Europe",
        "European Union",
    ],
)
def test_workday_location_triage_accept_examples(location):
    triage = workday._triage_location(location, ["Netherlands", "Remote", "Europe", "EU"], ["Canada", "India"])

    assert triage.decision == "accept"


@pytest.mark.parametrize(
    "location",
    [
        "",
        "Multiple Locations",
        "2 Locations",
        "Global",
        "Seattle, WA",
        "Oeiras, Portugal",
        "San Jose, Costa Rica",
        "Zug, Switzerland",
    ],
)
def test_workday_location_triage_manual_review_examples(location):
    triage = workday._triage_location(location, ["Netherlands", "Remote", "Europe", "EU"], ["Canada", "India"])

    assert triage.decision == "manual_review"


@pytest.mark.parametrize(
    "location",
    [
        "Noida, India",
        "Toronto, Canada",
        "Remote-Canada-British Columbia",
        "Remote - United States",
        "New York, United States",
        "San Francisco, California, United States",
    ],
)
def test_workday_location_triage_reject_examples(location):
    triage = workday._triage_location(
        location,
        ["Netherlands", "Remote", "Europe", "EU"],
        ["United States", "Canada", "India"],
    )

    assert triage.decision == "reject"


def test_select_employers_matches_keys_and_display_names():
    employers = {
        "thomson_reuters": {"name": "Thomson Reuters"},
        "salesforce": {"name": "Salesforce"},
        "nvidia": {"name": "NVIDIA"},
    }

    selected = workday._select_employers(employers, ["thomson reuters", "nvidia"])

    assert list(selected) == ["thomson_reuters", "nvidia"]


def test_manual_review_workday_jobs_are_not_ready_to_apply(tmp_path):
    conn = init_db(tmp_path / "applypilot.db")
    jobs = [
        {
            "title": "Data Engineer",
            "location": "Multiple Locations",
            "external_path": "/job/example",
            "employer_key": "example",
            "employer_name": "ExampleCo",
            "full_description": "Python data engineering role. " * 20,
            "apply_url": "https://example.com/apply",
            "location_decision": "manual_review",
            "location_reason": "ambiguous_location",
        }
    ]

    new, existing = workday.store_results(conn, jobs, {"example": _employer()})
    conn.execute(
        "UPDATE jobs SET fit_score = 8, tailored_resume_path = ? WHERE url = ?",
        (str(tmp_path / "resume.txt"), "https://example.com/apply"),
    )
    conn.commit()

    row = conn.execute("SELECT review_status, review_reason FROM jobs").fetchone()
    stats = get_stats(conn)

    assert (new, existing) == (1, 0)
    assert row["review_status"] == "manual_review"
    assert row["review_reason"] == "workday_location:ambiguous_location"
    assert stats["manual_review"] == 1
    assert stats["ready_to_apply"] == 0
    assert count_jobs_to_score(conn) == 0
    conn.close()


def test_workday_422_sources_return_no_rows_and_continue(monkeypatch, tmp_path, caplog):
    caplog.set_level(logging.ERROR)
    conn = init_db(tmp_path / "applypilot.db")
    employers = {
        "servicenow": _employer("ServiceNow"),
        "docusign": _employer("DocuSign"),
        "cisco": _employer("Cisco"),
    }

    def fake_workday_search(employer, _search_text, limit=20, offset=0):
        if employer["name"] in {"ServiceNow", "DocuSign"}:
            raise HTTPError(
                url="https://example.com",
                code=422,
                msg="Unprocessable Entity",
                hdrs=None,
                fp=None,
            )
        return {"total": 0, "jobPostings": []}

    monkeypatch.setattr(workday, "init_db", lambda: conn)
    monkeypatch.setattr(workday, "workday_search", fake_workday_search)

    result = workday.scrape_employers(
        "Data Engineer",
        employers,
        accept_locs=["Netherlands"],
        reject_locs=["United States", "Canada", "India"],
    )

    assert result == {"found": 0, "new": 0, "existing": 0}
    assert conn.execute("SELECT COUNT(*) FROM jobs").fetchone()[0] == 0
    assert "ServiceNow: API error at offset 0: HTTP Error 422" in caplog.text
    assert "DocuSign: API error at offset 0: HTTP Error 422" in caplog.text
    conn.close()
