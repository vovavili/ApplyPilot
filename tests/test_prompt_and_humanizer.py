import json
from pathlib import Path

import applypilot.config as app_config
from applypilot.database import init_db
from applypilot import humanizer
from applypilot.apply import prompt
from applypilot.scoring import cover_letter, pdf, tailor
from applypilot.scoring.validator import validate_cover_letter, validate_json_fields


def _profile() -> dict:
    return {
        "personal": {
            "full_name": "Test Candidate",
            "preferred_name": "Test",
            "email": "candidate@example.com",
            "phone": "+31000000000",
            "address": "Main Street 1",
            "city": "The Hague",
            "province_state": "South Holland",
            "country": "Netherlands",
            "postal_code": "2500AA",
            "linkedin_url": "https://www.linkedin.com/in/test/",
            "github_url": "",
            "portfolio_url": "",
            "website_url": "",
            "password": "test-password",
        },
        "work_authorization": {
            "legally_authorized_to_work": True,
            "require_sponsorship": False,
            "work_permit_type": "EU citizen",
        },
        "compensation": {
            "private_minimum": "123456",
            "target_anchor": "",
            "public_default_answer": "Negotiable",
            "salary_currency": "EUR",
            "if_required_numeric_and_no_range": "manual_review",
            "public_range_strategy": "manual_review",
        },
        "experience": {
            "years_of_experience_total": "4",
            "education_level": "Master's",
            "target_role": "Senior Data Engineer",
        },
        "availability": {"earliest_start_date": "Immediately"},
        "eeo_voluntary": {},
    }


def test_profile_summary_exposes_public_salary_not_private_threshold():
    summary = prompt._build_profile_summary(_profile())

    assert "Salary Public Answer: Negotiable" in summary
    assert "123456" not in summary
    assert "Private" not in summary


def test_build_prompt_keeps_private_salary_out_of_applicant_profile(monkeypatch, tmp_path):
    resume_txt = tmp_path / "tailored_resume.txt"
    resume_pdf = tmp_path / "tailored_resume.pdf"
    resume_txt.write_text("resume text", encoding="utf-8")
    resume_pdf.write_bytes(b"%PDF-1.4\n")

    monkeypatch.setattr(prompt.config, "load_profile", _profile)
    monkeypatch.setattr(prompt.config, "load_search_config", lambda: {"location": {"accept_patterns": ["The Hague"]}})
    monkeypatch.setattr(prompt.config, "APPLY_WORKER_DIR", tmp_path / "workers")
    monkeypatch.setattr(prompt, "_build_captcha_section", lambda: "CAPTCHA TEST SECTION")
    monkeypatch.setattr(prompt, "get_humanizer_prompt", lambda target: f"HUMANIZER:{target}")
    monkeypatch.setattr(app_config, "load_blocked_sso", lambda: [])

    built = prompt.build_prompt(
        job={
            "url": "https://example.com/job",
            "title": "Data Engineer",
            "site": "ExampleCo",
            "salary": "€130,000 - €150,000",
            "fit_score": 9,
            "tailored_resume_path": str(resume_txt),
        },
        tailored_resume="TAILORED RESUME TEXT",
    )

    applicant_profile = built.split("== APPLICANT PROFILE ==", 1)[1].split("== YOUR MISSION ==", 1)[0]
    assert "Salary Public Answer: Negotiable" in applicant_profile
    assert "123456" not in applicant_profile
    assert "Posted Salary: €130,000 - €150,000" in built
    assert "Private rejection threshold: 123456 EUR" in built
    assert "Never type, paste, say, paraphrase, or reveal this number" in built
    assert "HUMANIZER:screening" in built


def test_build_prompt_ignores_placeholder_application_url(monkeypatch, tmp_path):
    resume_txt = tmp_path / "tailored_resume.txt"
    resume_pdf = tmp_path / "tailored_resume.pdf"
    resume_txt.write_text("resume text", encoding="utf-8")
    resume_pdf.write_bytes(b"%PDF-1.4\n")

    monkeypatch.setattr(prompt.config, "load_profile", _profile)
    monkeypatch.setattr(prompt.config, "load_search_config", lambda: {"location": {"accept_patterns": ["The Hague"]}})
    monkeypatch.setattr(prompt.config, "APPLY_WORKER_DIR", tmp_path / "workers")
    monkeypatch.setattr(prompt, "_build_captcha_section", lambda: "CAPTCHA TEST SECTION")
    monkeypatch.setattr(prompt, "get_humanizer_prompt", lambda target: f"HUMANIZER:{target}")
    monkeypatch.setattr(app_config, "load_blocked_sso", lambda: [])

    built = prompt.build_prompt(
        job={
            "url": "https://example.com/job",
            "application_url": "None",
            "title": "Data Engineer",
            "site": "ExampleCo",
            "salary": "€130,000 - €150,000",
            "fit_score": 9,
            "tailored_resume_path": str(resume_txt),
        },
        tailored_resume="TAILORED RESUME TEXT",
    )

    assert "URL: https://example.com/job" in built
    assert "URL: None" not in built


def test_build_prompt_dry_run_uses_non_mutating_result_code(monkeypatch, tmp_path):
    resume_txt = tmp_path / "tailored_resume.txt"
    resume_pdf = tmp_path / "tailored_resume.pdf"
    resume_txt.write_text("resume text", encoding="utf-8")
    resume_pdf.write_bytes(b"%PDF-1.4\n")

    monkeypatch.setattr(prompt.config, "load_profile", _profile)
    monkeypatch.setattr(prompt.config, "load_search_config", lambda: {"location": {"accept_patterns": ["The Hague"]}})
    monkeypatch.setattr(prompt.config, "APPLY_WORKER_DIR", tmp_path / "workers")
    monkeypatch.setattr(prompt, "_build_captcha_section", lambda: "CAPTCHA TEST SECTION")
    monkeypatch.setattr(prompt, "get_humanizer_prompt", lambda target: f"HUMANIZER:{target}")
    monkeypatch.setattr(app_config, "load_blocked_sso", lambda: [])

    built = prompt.build_prompt(
        job={
            "url": "https://example.com/job",
            "application_url": "https://example.com/apply",
            "title": "Data Engineer",
            "site": "ExampleCo",
            "salary": "€130,000 - €150,000",
            "fit_score": 9,
            "tailored_resume_path": str(resume_txt),
        },
        tailored_resume="TAILORED RESUME TEXT",
        dry_run=True,
    )

    assert "RESULT:DRY_RUN" in built
    assert "Do NOT click the final Submit/Apply button" in built


def test_humanizer_is_opt_in_and_uses_configured_prompt(monkeypatch, tmp_path):
    prompt_path = tmp_path / "humanizer.md"
    prompt_path.write_text("Custom humanizer rules.", encoding="utf-8")
    monkeypatch.setattr(humanizer, "load_env", lambda: None)

    monkeypatch.delenv("APPLYPILOT_HUMANIZER", raising=False)
    assert humanizer.get_humanizer_prompt("screening") == ""

    monkeypatch.setenv("APPLYPILOT_HUMANIZER", "1")
    monkeypatch.setenv("APPLYPILOT_HUMANIZER_PROMPT_PATH", str(prompt_path))
    result = humanizer.get_humanizer_prompt("screening")

    assert "## HUMANIZER PASS" in result
    assert "Custom humanizer rules." in result
    assert "Never reveal a private minimum" in result
    assert str(Path(prompt_path)) not in result


def test_resume_humanizer_does_not_override_json_contract(monkeypatch):
    monkeypatch.setattr(tailor, "get_humanizer_prompt", lambda target: "\n\nHUMANIZER BLOCK")

    built = tailor._build_tailor_prompt(_profile())

    assert "HUMANIZER BLOCK" in built
    assert built.rfind("HUMANIZER BLOCK") < built.rfind("## OUTPUT: Return ONLY valid JSON")
    assert built.strip().endswith('"education":" | Master\'s"}')


def test_tailor_prompt_preserves_full_education_block(monkeypatch):
    monkeypatch.setattr(tailor, "get_humanizer_prompt", lambda target: "")
    profile = _profile()
    profile["education"] = [
        {
            "school": "Leiden University",
            "location": "Leiden, Netherlands",
            "degree": "MSc Computer Science",
            "dates": "Feb 2020 - Feb 2021",
        },
        {
            "school": "University of Helsinki",
            "location": "Helsinki, Finland",
            "degree": "Erasmus+ Computer Science",
            "dates": "Sep 2017 - Feb 2018",
        },
        {
            "school": "University of Latvia",
            "location": "Riga, Latvia",
            "degree": "BSc Computer Science",
            "dates": "Sep 2016 - Sep 2019",
        },
    ]

    built = tailor._build_tailor_prompt(profile)

    assert "Do not collapse three schools into one line" in built
    assert "Leiden University Leiden, Netherlands" in built
    assert "MSc Computer Science Feb 2020 - Feb 2021" in built
    assert "University of Helsinki Helsinki, Finland" in built
    assert '"education":"Leiden University Leiden, Netherlands\\nMSc Computer Science' in built


def test_assemble_resume_text_injects_location_and_full_education():
    profile = _profile()
    profile["education"] = [
        {
            "school": "Leiden University",
            "location": "Leiden, Netherlands",
            "degree": "MSc Computer Science",
            "dates": "Feb 2020 - Feb 2021",
        },
        {
            "school": "University of Helsinki",
            "location": "Helsinki, Finland",
            "degree": "Erasmus+ Computer Science",
            "dates": "Sep 2017 - Feb 2018",
        },
        {
            "school": "University of Latvia",
            "location": "Riga, Latvia",
            "degree": "BSc Computer Science",
            "dates": "Sep 2016 - Sep 2019",
        },
    ]
    data = {
        "title": "Data Engineer",
        "summary": "Built Python and SQL pipelines for analytics teams.",
        "skills": {"Languages": "Python, SQL"},
        "experience": [{"header": "Data Engineer at ExampleCo", "subtitle": "Python | 2024", "bullets": []}],
        "projects": [],
        "education": "Leiden University, University of Helsinki, University of Latvia | Master's",
    }

    text = tailor.assemble_resume_text(data, profile)

    assert "The Hague, Netherlands" in text
    assert "Leiden University Leiden, Netherlands" in text
    assert "MSc Computer Science Feb 2020 - Feb 2021" in text
    assert "University of Helsinki Helsinki, Finland" in text
    assert "Erasmus+ Computer Science Sep 2017 - Feb 2018" in text
    assert "Leiden University, University of Helsinki, University of Latvia | Master's" not in text


def test_cover_prompt_uses_structured_json_contract():
    built = cover_letter._build_cover_letter_prompt(_profile())

    assert "Return ONLY valid JSON" in built
    assert '"opening_work"' in built
    assert '"role_fit"' in built
    assert '"closing"' in built
    assert "170-285 words" in built
    assert 'Do not include "Dear Hiring Manager"' in built
    assert "BANNED WORDS" not in built


def test_cover_letter_validation_rejects_truncated_letters():
    result = validate_cover_letter(
        "Dear Hiring Manager,\n\nI built a Python data platform for hospital reporting. This",
        expected_signoff="Test",
    )

    assert not result["passed"]
    assert "Must end with sign-off name 'Test'" in result["errors"]
    assert any("Too short" in error for error in result["errors"])
    assert "Looks truncated; dangling final word 'this'" in result["errors"]


def test_cover_letter_validation_accepts_complete_letter():
    letter = """Dear Hiring Manager,

I built Python and SQL reporting pipelines for clinical teams that needed cleaner operational data and fewer manual reporting checks. The work combined Microsoft Fabric, Databricks, and Power BI, which is close to the reporting and analytics problems in this role. It also required the less glamorous parts of data work: clear ownership, repeatable deployment, and enough validation that stakeholders could trust the numbers.

At KPMG, I integrated API data into ESG reporting workflows and used Databricks Asset Bundles, Pydantic models, and Unity Catalog to make releases easier to review. At Medicine for Business, I deployed openEHR systems with Docker Compose and built Power BI dashboards for several hospitals. At Metyis/Adaptfy, I moved IoT workloads to Azure Data Factory, Databricks, and Cosmos DB, then added Great Expectations checks.

Your job description points to practical data engineering across ingestion, modeling, reporting, and quality. That is the kind of work I have done across healthcare, ESG, logistics, and finance, usually in settings where the system had to become understandable to analysts and business users after it shipped.

I would be glad to discuss how that mix of data engineering, reporting, and pragmatic delivery could help your team ship reliable analytics faster.

Test"""

    result = validate_cover_letter(letter, expected_signoff="Test")

    assert result["passed"]


def _good_cover_json() -> dict:
    return {
        "opening_work": (
            "I built Python and SQL pipelines that moved API data into Databricks and Power BI reporting "
            "for teams that needed cleaner operational data and fewer manual checks"
        ),
        "role_fit": (
            "The role maps well to my recent work across ingestion, modeling, validation, reporting, "
            "and production ownership in healthcare, ESG, logistics, and finance"
        ),
        "achievement_1": (
            "At KPMG, I integrated the Watershed API into ESG reporting workflows with Databricks Asset "
            "Bundles, Pydantic models, and Unity Catalog so releases were easier to review"
        ),
        "achievement_2": (
            "At Medicine for Business, I deployed openEHR systems with Docker Compose and built Microsoft "
            "Fabric and Power BI dashboards for hospital partners while keeping delivery practical"
        ),
        "company_reason": (
            "Your description points to practical ownership of data pipelines, validation, and reporting, "
            "which matches the systems I have shipped in healthcare, ESG, and logistics. I would bring "
            "that same bias toward clear pipelines, visible data quality, and maintainable handoffs"
        ),
        "closing": "I would be glad to walk through the tradeoffs, code, and delivery choices behind that work",
    }


def _fake_convert_to_pdf(path, *_args, **_kwargs):
    pdf_path = Path(path).with_suffix(".pdf")
    pdf_path.write_bytes(b"%PDF-1.4\n")
    return pdf_path


def test_structured_cover_json_assembles_complete_letter():
    letter = cover_letter._assemble_cover_letter(
        _good_cover_json(),
        {
            "title": "Data Engineer",
            "site": "ExampleCo",
            "location": "Amsterdam",
        },
        _profile(),
    )

    assert letter.startswith("Dear Hiring Manager,")
    assert letter.endswith("\nTest")
    assert "Data Engineer" in letter
    assert "ExampleCo" in letter
    assert validate_cover_letter(letter, expected_signoff="Test")["passed"]
    quality = cover_letter._cover_content_quality(letter)
    assert quality["passed"]
    assert 170 <= quality["word_count"] <= 285


def test_cover_uses_inferred_company_for_job_board_titles():
    letter = cover_letter._assemble_cover_letter(
        _good_cover_json(),
        {
            "title": "Data Engineer | Cmotions",
            "site": "linkedin",
            "location": "Breda, Netherlands",
        },
        _profile(),
    )

    assert "Data Engineer role at Cmotions" in letter
    assert "linkedin" not in letter.casefold()


def test_almost_json_cover_response_can_be_recovered():
    raw = json.dumps(_good_cover_json(), indent=2).removesuffix("}")
    data = cover_letter._extract_cover_json(raw)

    assert data == _good_cover_json()
    letter = cover_letter._assemble_cover_letter(data, {"title": "Data Engineer", "site": "ExampleCo"}, _profile())
    assert validate_cover_letter(letter, expected_signoff="Test")["passed"]


def test_good_structured_response_produces_non_fallback_cover(monkeypatch):
    class FakeClient:
        model = "fake-model"

        def chat(self, *_args, **_kwargs):
            return json.dumps(_good_cover_json())

    monkeypatch.setattr(cover_letter, "get_client", lambda: FakeClient())

    letter, report = cover_letter.generate_cover_letter(
        "Tailored resume with Python, SQL, Databricks, Microsoft Fabric, and Power BI.",
        {
            "title": "Data Engineer",
            "site": "ExampleCo",
            "location": "Amsterdam",
            "full_description": "Python SQL Databricks data pipelines reporting",
        },
        _profile(),
        max_retries=0,
    )

    assert "ExampleCo" in letter
    assert report["status"] == "approved"
    assert report["fallback"] is False
    assert report["generator"] == "structured_json"
    assert report["content_quality"]["passed"]


def test_cover_generation_uses_cover_model_override(monkeypatch):
    class FakeClient:
        model = "fake-cover-model"

        def chat(self, *_args, **_kwargs):
            return json.dumps(_good_cover_json())

    requested_models = []

    def fake_make_client(model=None):
        requested_models.append(model)
        return FakeClient()

    monkeypatch.setenv("COVER_LLM_MODEL", "gemini-3.1-pro-preview")
    monkeypatch.setattr(cover_letter, "make_client", fake_make_client)

    _letter, report = cover_letter.generate_cover_letter(
        "Tailored resume with Python and SQL.",
        {
            "title": "Data Engineer",
            "site": "ExampleCo",
            "location": "Amsterdam",
            "full_description": "Python SQL Databricks data pipelines reporting",
        },
        _profile(),
        max_retries=0,
    )

    assert requested_models == ["gemini-3.1-pro-preview"]
    assert report["cover_model_override"] is True
    assert report["model"] == "fake-cover-model"


def test_cover_generation_routes_claude_model_override_to_anthropic(monkeypatch):
    class FakeClient:
        model = "claude-opus-4-7"

        def chat(self, *_args, **_kwargs):
            return json.dumps(_good_cover_json())

    requested_models = []

    def fake_make_anthropic_client(model=None):
        requested_models.append(model)
        return FakeClient()

    monkeypatch.setenv("COVER_LLM_MODEL", "claude-opus-4-7")
    monkeypatch.setattr(cover_letter, "make_anthropic_client", fake_make_anthropic_client)

    _letter, report = cover_letter.generate_cover_letter(
        "Tailored resume with Python and SQL.",
        {
            "title": "Data Engineer",
            "site": "ExampleCo",
            "location": "Amsterdam",
            "full_description": "Python SQL Databricks data pipelines reporting",
        },
        _profile(),
        max_retries=0,
    )

    assert requested_models == ["claude-opus-4-7"]
    assert report["cover_model_override"] is True
    assert report["model"] == "claude-opus-4-7"


def test_too_brief_structured_cover_retries_then_falls_back(monkeypatch):
    class FakeClient:
        calls = 0
        model = "fake-model"

        def chat(self, *_args, **_kwargs):
            self.calls += 1
            return json.dumps(
                {
                    "opening_work": "I build Python data pipelines",
                    "role_fit": "The role needs SQL reporting work",
                    "achievement_1": "At KPMG, I worked on ESG reporting",
                    "achievement_2": "At Medicine for Business, I built dashboards",
                    "company_reason": "Your team needs practical data ownership",
                    "closing": "I would be glad to discuss the work",
                }
            )

    client = FakeClient()
    monkeypatch.delenv("COVER_LLM_MODEL", raising=False)
    monkeypatch.delenv("COVER_LLM_PROVIDER", raising=False)
    monkeypatch.setattr(cover_letter, "get_client", lambda: client)

    letter, report = cover_letter.generate_cover_letter(
        "Tailored resume with Python and SQL.",
        {
            "title": "Data Engineer",
            "site": "ExampleCo",
            "location": "Amsterdam",
            "full_description": "Python SQL Databricks data pipelines reporting",
        },
        _profile(),
        max_retries=1,
    )

    assert client.calls == 2
    assert report["fallback"] is True
    assert report["fallback_reason"] == "llm_failed_validation"
    assert any("Too brief" in error for error in report["failed_attempts"][0]["errors"])
    assert report["content_quality"]["passed"]
    assert len(letter.split()) >= 170


def test_cover_letter_validation_catches_banned_words_in_strict_mode():
    letter = """Dear Hiring Manager,

For the Data Engineer role at ExampleCo, I am excited to bring Python and SQL data work to a team that needs production reporting. I built Databricks pipelines and Power BI dashboards for teams that needed cleaner operational data.

At KPMG, I integrated API data into ESG reporting workflows with Pydantic models and Unity Catalog. At Medicine for Business, I deployed openEHR systems with Docker Compose and built Microsoft Fabric reporting for hospital partners.

Your description points to hands-on data ownership, validation, and reporting. I would be glad to walk through the tradeoffs and the code behind that work.

Test"""

    result = validate_cover_letter(letter, mode="strict", expected_signoff="Test")

    assert not result["passed"]
    assert any("Banned words" in error for error in result["errors"])


def test_tailor_resume_reports_invalid_json(monkeypatch):
    class FakeClient:
        def chat(self, *_args, **_kwargs):
            return "Here is a polished resume in prose, not JSON."

    monkeypatch.setattr(tailor, "get_client", lambda: FakeClient())
    monkeypatch.setattr(tailor, "get_humanizer_prompt", lambda target: "")

    text, report = tailor.tailor_resume(
        "Base resume",
        {
            "title": "Data Engineer",
            "site": "ExampleCo",
            "location": "Amsterdam",
            "full_description": "Python data pipelines",
        },
        _profile(),
        max_retries=0,
    )

    assert text == ""
    assert report["status"] == "invalid_json"
    assert report["parse_errors"][0]["raw_excerpt"] == "Here is a polished resume in prose, not JSON."


def test_tailor_generation_uses_tailor_model_override(monkeypatch):
    class FakeClient:
        model = "fake-tailor-model"

        def chat(self, *_args, **_kwargs):
            return json.dumps(
                {
                    "title": "Data Engineer",
                    "summary": "Built Python and SQL pipelines for analytics teams.",
                    "skills": {"Languages": "Python, SQL"},
                    "experience": [
                        {
                            "header": "Data Engineer at ExampleCo",
                            "subtitle": "Python | 2024",
                            "bullets": ["Built ETL pipelines."],
                        }
                    ],
                    "projects": [],
                    "education": "Master's",
                }
            )

    requested_models = []

    def fake_make_client(model=None):
        requested_models.append(model)
        return FakeClient()

    monkeypatch.setenv("TAILOR_LLM_MODEL", "gemini-3.1-pro-preview")
    monkeypatch.setattr(tailor, "make_client", fake_make_client)
    monkeypatch.setattr(tailor, "get_humanizer_prompt", lambda target: "")

    text, report = tailor.tailor_resume(
        "Base resume",
        {
            "title": "Data Engineer",
            "site": "ExampleCo",
            "location": "Amsterdam",
            "full_description": "Python data pipelines",
        },
        _profile(),
        max_retries=0,
        validation_mode="lenient",
    )

    assert requested_models == ["gemini-3.1-pro-preview"]
    assert report["tailor_model_override"] is True
    assert report["model"] == "fake-tailor-model"
    assert "Data Engineer" in text


def test_tailor_generation_routes_claude_model_override_to_anthropic(monkeypatch):
    class FakeClient:
        model = "claude-opus-4-7"

        def chat(self, *_args, **_kwargs):
            return json.dumps(
                {
                    "title": "Data Engineer",
                    "summary": "Built Python and SQL pipelines for analytics teams.",
                    "skills": {"Languages": "Python, SQL"},
                    "experience": [
                        {
                            "header": "Data Engineer at ExampleCo",
                            "subtitle": "Python | 2024",
                            "bullets": ["Built ETL pipelines."],
                        }
                    ],
                    "projects": [],
                    "education": "Master's",
                }
            )

    requested_models = []

    def fake_make_anthropic_client(model=None):
        requested_models.append(model)
        return FakeClient()

    monkeypatch.setenv("TAILOR_LLM_MODEL", "claude-opus-4-7")
    monkeypatch.setattr(tailor, "make_anthropic_client", fake_make_anthropic_client)
    monkeypatch.setattr(tailor, "get_humanizer_prompt", lambda target: "")

    _text, report = tailor.tailor_resume(
        "Base resume",
        {
            "title": "Data Engineer",
            "site": "ExampleCo",
            "location": "Amsterdam",
            "full_description": "Python data pipelines",
        },
        _profile(),
        max_retries=0,
        validation_mode="lenient",
    )

    assert requested_models == ["claude-opus-4-7"]
    assert report["tailor_model_override"] is True
    assert report["model"] == "claude-opus-4-7"


def test_run_tailoring_invalid_json_does_not_create_empty_resume(monkeypatch, tmp_path):
    class FakeClient:
        def chat(self, *_args, **_kwargs):
            return "Here is a polished resume in prose, not JSON."

    conn = init_db(tmp_path / "applypilot.db")
    resume_path = tmp_path / "resume.txt"
    tailored_dir = tmp_path / "tailored"
    resume_path.write_text("Base resume", encoding="utf-8")
    conn.execute(
        """
        INSERT INTO jobs (
            url, title, site, discovered_at, full_description, detail_scraped_at, fit_score
        )
        VALUES (?, ?, ?, ?, ?, ?, ?)
        """,
        (
            "https://example.com/job",
            "Data Engineer",
            "ExampleCo",
            "2026-05-25T18:00:00+00:00",
            "Python data pipelines",
            "2026-05-25T18:00:00+00:00",
            8,
        ),
    )
    conn.commit()

    monkeypatch.setattr(tailor, "get_connection", lambda: conn)
    monkeypatch.setattr(tailor, "get_client", lambda: FakeClient())
    monkeypatch.setattr(tailor, "get_humanizer_prompt", lambda target: "")
    monkeypatch.setattr(tailor, "load_profile", _profile)
    monkeypatch.setattr(tailor, "RESUME_PATH", resume_path)
    monkeypatch.setattr(tailor, "TAILORED_DIR", tailored_dir)

    result = tailor.run_tailoring(limit=1, validation_mode="lenient")

    assert result["approved"] == 0
    assert result["failed"] == 1
    assert not (tailored_dir / "ExampleCo_Data_Engineer.txt").exists()
    assert all(path.stat().st_size > 0 for path in tailored_dir.glob("*"))
    report = next(tailored_dir.glob("*_REPORT.json"))
    assert '"status": "invalid_json"' in report.read_text(encoding="utf-8")
    assert conn.execute("SELECT tailored_resume_path FROM jobs").fetchone()[0] is None
    assert conn.execute("SELECT tailor_attempts FROM jobs").fetchone()[0] == 1


def test_job_file_prefix_uses_url_digest_to_avoid_collisions():
    first = {
        "url": "https://example.com/jobs/1",
        "title": "Data Engineer",
        "site": "ExampleCo",
    }
    second = {
        "url": "https://example.com/jobs/2",
        "title": "Data Engineer",
        "site": "ExampleCo",
    }

    assert tailor._job_file_prefix(first) != tailor._job_file_prefix(second)
    assert cover_letter._job_file_prefix(first) != cover_letter._job_file_prefix(second)


def test_tailor_json_validation_allows_missing_projects():
    data = {
        "title": "Data Engineer",
        "summary": "Built Python and SQL pipelines for analytics teams.",
        "skills": {"Languages": "Python, SQL"},
        "experience": [{"header": "Data Engineer at ExampleCo", "bullets": ["Built ETL pipelines."]}],
        "education": "Master's",
    }

    result = validate_json_fields(data, _profile(), mode="lenient")

    assert result["passed"]


def test_tailor_json_validation_allows_watchlist_skill_when_profile_allows_it():
    profile = _profile()
    profile["skills_boundary"] = {"programming_languages": ["Python", "SQL", "Rust", "Scala"]}
    data = {
        "title": "Data Engineer",
        "summary": "Built Python and SQL pipelines for analytics teams.",
        "skills": {"Languages": "Python, SQL, Rust, Scala"},
        "experience": [{"header": "Data Engineer at ExampleCo", "bullets": ["Built ETL pipelines."]}],
        "education": "Master's",
    }

    result = validate_json_fields(data, profile, mode="normal")

    assert result["passed"]


def test_tailor_json_validation_accepts_preserved_schools_with_degree_labels():
    profile = _profile()
    profile["resume_facts"] = {"preserved_school": "Leiden University, University of Helsinki, University of Latvia"}
    data = {
        "title": "Data Engineer",
        "summary": "Built Python and SQL pipelines for analytics teams.",
        "skills": {"Languages": "Python, SQL"},
        "experience": [{"header": "Data Engineer at ExampleCo", "bullets": ["Built ETL pipelines."]}],
        "education": (
            "Leiden University (MSc Computer Science), "
            "University of Helsinki (Erasmus+ Computer Science), "
            "University of Latvia (BSc Computer Science)"
        ),
    }

    result = validate_json_fields(data, profile, mode="normal")

    assert result["passed"]


def test_run_cover_letters_uses_safe_fallback_after_truncated_output(monkeypatch, tmp_path):
    class FakeClient:
        calls = 0

        def chat(self, *_args, **_kwargs):
            self.calls += 1
            return "Dear Hiring Manager,\n\nI built Python data pipelines. This"

    client = FakeClient()
    conn = init_db(tmp_path / "applypilot.db")
    resume_path = tmp_path / "resume.txt"
    cover_dir = tmp_path / "cover"
    tailored_path = tmp_path / "tailored.txt"
    resume_path.write_text("Base resume", encoding="utf-8")
    tailored_path.write_text("Tailored resume", encoding="utf-8")
    conn.execute(
        """
        INSERT INTO jobs (
            url, title, site, discovered_at, full_description, detail_scraped_at,
            fit_score, tailored_resume_path, location
        )
        VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?)
        """,
        (
            "https://example.com/job",
            "Data Engineer",
            "ExampleCo",
            "2026-05-25T18:00:00+00:00",
            "Python data pipelines",
            "2026-05-25T18:00:00+00:00",
            8,
            str(tailored_path),
            "Amsterdam, Netherlands",
        ),
    )
    conn.commit()
    converted = {}

    def fake_convert_to_pdf(path, *_args, **kwargs):
        converted["path"] = Path(path)
        converted["job"] = kwargs.get("job")
        converted["profile"] = kwargs.get("profile")
        return _fake_convert_to_pdf(path)

    monkeypatch.setattr(cover_letter, "get_connection", lambda: conn)
    monkeypatch.setattr(cover_letter, "get_client", lambda: client)
    monkeypatch.setattr(cover_letter, "load_profile", _profile)
    monkeypatch.setattr(cover_letter, "RESUME_PATH", resume_path)
    monkeypatch.setattr(cover_letter, "COVER_LETTER_DIR", cover_dir)
    monkeypatch.setattr(pdf, "convert_to_pdf", fake_convert_to_pdf)

    result = cover_letter.run_cover_letters(limit=1, validation_mode="normal")

    assert result["generated"] == 1
    assert result["errors"] == 0
    assert result["fallback"] == 1
    row = conn.execute("SELECT cover_letter_path, cover_attempts FROM jobs").fetchone()
    assert row["cover_letter_path"]
    assert row["cover_attempts"] == 1
    report = next(cover_dir.glob("*_CL_REPORT.json")).read_text(encoding="utf-8")
    assert '"status": "approved"' in report
    assert '"fallback": true' in report
    assert '"raw_excerpt": "Dear Hiring Manager' in report
    assert client.calls == 2


def test_run_cover_letters_uses_safe_fallback_after_llm_response_error(monkeypatch, tmp_path):
    class FakeClient:
        calls = 0

        def chat(self, *_args, **_kwargs):
            self.calls += 1
            raise KeyError("content")

    client = FakeClient()
    conn = init_db(tmp_path / "applypilot.db")
    resume_path = tmp_path / "resume.txt"
    cover_dir = tmp_path / "cover"
    tailored_path = tmp_path / "tailored.txt"
    resume_path.write_text("Base resume", encoding="utf-8")
    tailored_path.write_text("Tailored resume", encoding="utf-8")
    conn.execute(
        """
        INSERT INTO jobs (
            url, title, site, discovered_at, full_description, detail_scraped_at,
            fit_score, tailored_resume_path, location
        )
        VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?)
        """,
        (
            "https://example.com/job",
            "Python Developer",
            "Ciena",
            "2026-05-25T18:00:00+00:00",
            "Python data pipelines",
            "2026-05-25T18:00:00+00:00",
            8,
            str(tailored_path),
            "Amsterdam, Netherlands",
        ),
    )
    conn.commit()
    converted = {}

    def fake_convert_to_pdf(path, *_args, **kwargs):
        converted["path"] = Path(path)
        converted["job"] = kwargs.get("job")
        converted["profile"] = kwargs.get("profile")
        return _fake_convert_to_pdf(path)

    monkeypatch.setattr(cover_letter, "get_connection", lambda: conn)
    monkeypatch.setattr(cover_letter, "get_client", lambda: client)
    monkeypatch.setattr(cover_letter, "load_profile", _profile)
    monkeypatch.setattr(cover_letter, "RESUME_PATH", resume_path)
    monkeypatch.setattr(cover_letter, "COVER_LETTER_DIR", cover_dir)
    monkeypatch.setattr(pdf, "convert_to_pdf", fake_convert_to_pdf)

    result = cover_letter.run_cover_letters(limit=1, validation_mode="normal")

    assert result["generated"] == 1
    assert result["errors"] == 0
    assert result["fallback"] == 1
    assert client.calls == 2
    report = next(cover_dir.glob("*_CL_REPORT.json")).read_text(encoding="utf-8")
    assert '"fallback": true' in report
    assert '"fallback_reason": "llm_error"' in report
    assert '"llm_errors"' in report


def test_run_cover_letters_uses_tailored_resume_context(monkeypatch, tmp_path):
    class FakeClient:
        model = "fake-model"

        def __init__(self):
            self.messages = None

        def chat(self, messages, **_kwargs):
            self.messages = messages
            return json.dumps(_good_cover_json())

    client = FakeClient()
    conn = init_db(tmp_path / "applypilot.db")
    resume_path = tmp_path / "resume.txt"
    cover_dir = tmp_path / "cover"
    tailored_path = tmp_path / "tailored.txt"
    resume_path.write_text("BASE_ONLY resume", encoding="utf-8")
    tailored_path.write_text("TAILORED_ONLY resume with Databricks and Power BI", encoding="utf-8")
    conn.execute(
        """
        INSERT INTO jobs (
            url, title, site, discovered_at, full_description, detail_scraped_at,
            fit_score, tailored_resume_path, location
        )
        VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?)
        """,
        (
            "https://example.com/job",
            "Data Engineer",
            "ExampleCo",
            "2026-05-25T18:00:00+00:00",
            "Python SQL Databricks data pipelines reporting",
            "2026-05-25T18:00:00+00:00",
            8,
            str(tailored_path),
            "Amsterdam, Netherlands",
        ),
    )
    conn.commit()
    converted = {}

    def fake_convert_to_pdf(path, *_args, **kwargs):
        converted["path"] = Path(path)
        converted["job"] = kwargs.get("job")
        converted["profile"] = kwargs.get("profile")
        return _fake_convert_to_pdf(path)

    monkeypatch.setattr(cover_letter, "get_connection", lambda: conn)
    monkeypatch.setattr(cover_letter, "get_client", lambda: client)
    monkeypatch.setattr(cover_letter, "load_profile", _profile)
    monkeypatch.setattr(cover_letter, "RESUME_PATH", resume_path)
    monkeypatch.setattr(cover_letter, "COVER_LETTER_DIR", cover_dir)
    monkeypatch.setattr(pdf, "convert_to_pdf", fake_convert_to_pdf)

    result = cover_letter.run_cover_letters(limit=1, validation_mode="normal")

    user_prompt = client.messages[1]["content"]
    assert result["generated"] == 1
    assert result["fallback"] == 0
    assert "TAILORED_ONLY" in user_prompt
    assert "BASE_ONLY" not in user_prompt
    assert converted["job"]["title"] == "Data Engineer"
    assert converted["profile"]["personal"]["full_name"] == "Test Candidate"
