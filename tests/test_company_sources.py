import logging
import sys
import types

from applypilot import pipeline
from applypilot.database import get_stats, init_db
from applypilot.discovery import company_sources
from applypilot.scoring.selection import count_jobs_to_score


def _company(platform="greenhouse", **overrides):
    company = {"name": "ExampleCo", "platform": platform}
    if platform == "greenhouse":
        company["board"] = "example"
    elif platform == "lever":
        company["account"] = "example"
    elif platform == "workday":
        company["employer_key"] = "example"
    company.update(overrides)
    return company


def _accepted_job(company="ExampleCo", url="https://example.com/job"):
    return {
        "company": company,
        "title": "Data Engineer",
        "url": url,
        "application_url": url,
        "location": "Amsterdam, Netherlands",
        "full_description": "Python data engineering role. " * 20,
        "location_decision": "accept",
        "location_reason": "accepted_nl_or_configured",
    }


def test_load_company_sources_merges_user_registry_and_user_wins(monkeypatch, tmp_path):
    package_dir = tmp_path / "package"
    app_dir = tmp_path / "app"
    package_dir.mkdir()
    app_dir.mkdir()
    (package_dir / "company_sources.yaml").write_text(
        """
companies:
  shared:
    name: "Package Shared"
    platform: "greenhouse"
    board: "package"
  packaged_only:
    name: "Packaged Only"
    platform: "lever"
    account: "packaged"
""",
        encoding="utf-8",
    )
    (app_dir / "company_sources.yaml").write_text(
        """
companies:
  shared:
    name: "User Shared"
    platform: "greenhouse"
    board: "user"
  user_only:
    name: "User Only"
    platform: "uber"
""",
        encoding="utf-8",
    )
    monkeypatch.setattr(company_sources, "CONFIG_DIR", package_dir)
    monkeypatch.setattr(company_sources.config, "APP_DIR", app_dir)

    companies = company_sources.load_company_sources()

    assert set(companies) == {"shared", "packaged_only", "user_only"}
    assert companies["shared"]["name"] == "User Shared"
    assert companies["shared"]["board"] == "user"


def test_validate_company_sources_warns_and_skips_malformed(caplog):
    caplog.set_level(logging.WARNING)

    valid = company_sources._validate_companies(
        {
            "good": _company(),
            "bad": {"name": "Broken"},
            "not_mapping": "lol nope",
        },
        source="test",
    )

    assert list(valid) == ["good"]
    assert "Skipping malformed priority company bad" in caplog.text
    assert "Skipping malformed priority company not_mapping" in caplog.text


def test_select_company_sources_warns_for_unknown_entries(caplog):
    caplog.set_level(logging.WARNING)

    selected = company_sources._select_companies({"stripe": _company(name="Stripe")}, ["stripe", "missing"])

    assert list(selected) == ["stripe"]
    assert "Configured priority company sources not found: missing" in caplog.text


def test_greenhouse_adapter_maps_jobs(monkeypatch):
    monkeypatch.setattr(
        company_sources,
        "_http_get_json",
        lambda _url: {
            "jobs": [
                {
                    "title": "Senior Data Engineer",
                    "absolute_url": "https://stripe.com/jobs/123",
                    "content": "<p>Build Python data pipelines.</p>",
                    "location": {"name": "Amsterdam, Netherlands"},
                },
                {
                    "title": "Sales Manager",
                    "absolute_url": "https://stripe.com/jobs/456",
                    "content": "<p>Sales role.</p>",
                    "location": {"name": "Amsterdam, Netherlands"},
                },
            ]
        },
    )

    jobs = company_sources._scrape_greenhouse(
        "stripe",
        _company(name="Stripe", board="stripe"),
        ["Data Engineer"],
        ["Netherlands"],
        ["United States"],
        "recall_first",
    )

    assert len(jobs) == 1
    assert jobs[0]["company"] == "Stripe"
    assert jobs[0]["title"] == "Senior Data Engineer"
    assert jobs[0]["full_description"] == "Build Python data pipelines."
    assert jobs[0]["location_decision"] == "accept"


def test_lever_adapter_maps_jobs(monkeypatch):
    monkeypatch.setattr(
        company_sources,
        "_http_get_json",
        lambda _url: [
            {
                "text": "Analytics Engineer",
                "hostedUrl": "https://jobs.lever.co/spotify/123",
                "applyUrl": "https://jobs.lever.co/spotify/123/apply",
                "descriptionPlain": "Own SQL models and dashboards.",
                "additionalPlain": "Python is useful.",
                "categories": {"location": "Remote - EMEA"},
                "lists": [{"text": "What you'll do", "content": "<p>Build metrics.</p>"}],
            }
        ],
    )

    jobs = company_sources._scrape_lever(
        "spotify",
        _company("lever", name="Spotify", account="spotify"),
        ["Analytics Engineer"],
        ["Europe", "Remote"],
        ["United States"],
        "recall_first",
    )

    assert len(jobs) == 1
    assert jobs[0]["company"] == "Spotify"
    assert jobs[0]["application_url"].endswith("/apply")
    assert "Build metrics." in jobs[0]["full_description"]
    assert jobs[0]["location_decision"] == "accept"


def test_netflix_static_adapter_parses_positions(monkeypatch):
    monkeypatch.setattr(company_sources, "_fetch_netflix_description", lambda _url: "Netflix data platform role.")
    html = """
    <script>
    window.__DATA__ = {"positions": [
      {
        "id": 790315367185,
        "name": "Analytics Engineer 5 - Legal",
        "posting_name": "Analytics Engineer 5 - Legal",
        "location": "Amsterdam,Netherlands",
        "department": "Data & Insights",
        "canonicalPositionUrl": "https://explore.jobs.netflix.net/careers/job/790315367185",
        "job_description": ""
      }
    ]};
    </script>
    """

    jobs = company_sources._scrape_netflix_static(
        "netflix",
        _company("netflix", name="Netflix"),
        html,
        ["Analytics Engineer"],
        ["Netherlands"],
        ["United States"],
        "recall_first",
    )

    assert len(jobs) == 1
    assert jobs[0]["company"] == "Netflix"
    assert jobs[0]["location"] == "Amsterdam, Netherlands"
    assert jobs[0]["full_description"] == "Netflix data platform role."


def test_uber_static_adapter_parses_job_links():
    html = """
    <div class="job-row">
      <a href="/us/en/careers/list/152407/">Data Engineer II</a>
      <span>Amsterdam, Netherlands</span>
    </div>
    """

    jobs = company_sources._scrape_uber_static(
        "uber",
        _company("uber", name="Uber"),
        html,
        ["Data Engineer"],
        ["Netherlands"],
        ["United States"],
        "recall_first",
    )

    assert len(jobs) == 1
    assert jobs[0]["company"] == "Uber"
    assert jobs[0]["url"] == "https://www.uber.com/us/en/careers/list/152407/"
    assert jobs[0]["location"] == "Amsterdam, Netherlands"


def test_uber_api_adapter_trusts_location_filtered_search(monkeypatch):
    requests = []

    def fake_post_json(_url, payload, *, referer=None):
        requests.append((payload, referer))
        return {
            "status": "success",
            "data": {
                "totalResults": {"low": 1},
                "results": [
                    {
                        "id": 159080,
                        "title": "Software Engineer II, Python",
                        "description": "Amsterdam engineering role with data-heavy systems.",
                        "location": {"city": "Amsterdam", "country": "NLD", "countryName": "Netherlands"},
                        "allLocations": [
                            {"city": "Amsterdam", "country": "NLD", "countryName": "Netherlands"},
                        ],
                    }
                ],
            },
        }

    monkeypatch.setattr(company_sources, "_http_post_json", fake_post_json)

    jobs = company_sources._scrape_uber_api(
        "uber",
        _company("uber", name="Uber"),
        "Data Engineer",
        "Amsterdam",
        ["Netherlands"],
        ["United States"],
        "recall_first",
    )

    assert len(jobs) == 1
    assert jobs[0]["title"] == "Software Engineer II, Python"
    assert jobs[0]["url"] == "https://www.uber.com/us/en/careers/list/159080/"
    assert jobs[0]["location"] == "Amsterdam, Netherlands"
    assert requests[0][0]["params"]["location"] == [
        {"city": "Amsterdam", "country": "NLD", "countryName": "Netherlands"}
    ]


def test_query_matching_handles_reordered_title_tokens():
    assert company_sources._matches_queries(["Software Engineer, Data Platform"], ["Data Platform Engineer"])
    assert not company_sources._matches_queries(["Product Manager, Data Platform"], ["Data Platform Engineer"])
    assert not company_sources._matches_queries(
        ["Android Engineer", "This team works with product data."],
        ["Data Engineer"],
    )
    assert not company_sources._matches_queries(
        ["Engineering Manager", "You will work with Data Engineer teams."],
        ["Data Engineer"],
    )
    assert not company_sources._matches_queries(["Data Center Engineer"], ["Data Engineer"])


def test_run_priority_company_discovery_stores_jobs_and_keeps_source_failures_nonfatal(monkeypatch, tmp_path):
    conn = init_db(tmp_path / "applypilot.db")
    monkeypatch.setattr(company_sources, "init_db", lambda: conn)
    monkeypatch.setattr(company_sources, "get_connection", lambda: conn)
    monkeypatch.setattr(
        company_sources.config,
        "load_search_config",
        lambda: {
            "priority_company_sources_enabled": True,
            "priority_company_sources": ["good", "bad"],
            "queries": [{"query": "Data Engineer", "tier": 1}],
            "location_accept": ["Netherlands"],
            "location_reject_non_remote": ["United States"],
        },
    )

    def fake_scrape_source(key, *_args, **_kwargs):
        if key == "bad":
            raise RuntimeError("source exploded")
        return [_accepted_job(company="GoodCo")]

    monkeypatch.setattr(company_sources, "_scrape_source", fake_scrape_source)

    result = company_sources.run_priority_company_discovery(
        companies={
            "good": _company(name="GoodCo"),
            "bad": _company(name="BadCo"),
        }
    )

    row = conn.execute("SELECT site, strategy FROM jobs").fetchone()
    assert result["errors"] == 1
    assert result["new"] == 1
    assert row["site"] == "GoodCo"
    assert row["strategy"] == "priority_company"
    conn.close()


def test_manual_review_priority_company_jobs_are_not_ready_to_apply(tmp_path):
    conn = init_db(tmp_path / "applypilot.db")
    new, existing = company_sources.store_priority_company_jobs(
        conn,
        [
            {
                **_accepted_job(),
                "location": "Multiple Locations",
                "location_decision": "manual_review",
                "location_reason": "ambiguous_location",
            }
        ],
    )
    conn.execute(
        "UPDATE jobs SET fit_score = 8, tailored_resume_path = ? WHERE url = ?",
        (str(tmp_path / "resume.txt"), "https://example.com/job"),
    )
    conn.commit()

    row = conn.execute("SELECT review_status, review_reason FROM jobs").fetchone()
    stats = get_stats(conn)

    assert (new, existing) == (1, 0)
    assert row["review_status"] == "manual_review"
    assert row["review_reason"] == "priority_company_location:ambiguous_location"
    assert stats["ready_to_apply"] == 0
    conn.close()


def test_manual_review_priority_company_jobs_are_not_scored(tmp_path):
    conn = init_db(tmp_path / "applypilot.db")
    bad_locations = ["Seattle, WA", "Toronto", "Shanghai", "Zug, Switzerland"]
    jobs = [
        {
            **_accepted_job(url=f"https://example.com/job/{i}"),
            "location": location,
            "location_decision": "manual_review",
            "location_reason": "unmatched_location",
        }
        for i, location in enumerate(bad_locations)
    ]

    new, existing = company_sources.store_priority_company_jobs(conn, jobs)

    assert (new, existing) == (4, 0)
    assert conn.execute("SELECT COUNT(*) FROM jobs WHERE review_status = 'manual_review'").fetchone()[0] == 4
    assert count_jobs_to_score(conn) == 0
    conn.close()


def test_pipeline_discover_runs_priority_company_sources_and_reports_partial(monkeypatch):
    jobspy = types.ModuleType("applypilot.discovery.jobspy")
    jobspy.run_discovery = lambda: None
    workday = types.ModuleType("applypilot.discovery.workday")
    workday.run_workday_discovery = lambda workers=1: None
    smartextract = types.ModuleType("applypilot.discovery.smartextract")
    smartextract.run_smart_extract = lambda workers=1: None
    monkeypatch.setitem(sys.modules, "applypilot.discovery.jobspy", jobspy)
    monkeypatch.setitem(sys.modules, "applypilot.discovery.workday", workday)
    monkeypatch.setitem(sys.modules, "applypilot.discovery.smartextract", smartextract)
    monkeypatch.setattr(
        company_sources,
        "run_priority_company_discovery",
        lambda workers=1: {"found": 0, "new": 0, "existing": 0, "sources": 1, "errors": 1},
    )

    result = pipeline._run_discover()

    assert result["jobspy"] == "ok"
    assert result["workday"] == "ok"
    assert result["company_sources"] == "error: 1 priority company source(s) failed"
    assert result["smartextract"] == "ok"
