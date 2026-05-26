from pathlib import Path

from applypilot import database
from applypilot.database import init_db
from applypilot.scoring import pdf
from applypilot.scoring.latex_cover_letter import build_latex_cover_letter


RESUME_TEXT = """Test Candidate
Senior Data Engineer
The Hague, Netherlands
candidate@example.com | https://www.linkedin.com/in/test/

SUMMARY
Senior data engineer building Python and Databricks pipelines.

TECHNICAL SKILLS
Languages: Python, SQL
Data Engineering: Databricks, Airflow

EXPERIENCE
KPMG, Amsterdam
Senior Data Engineer | Sep 2025 - Jan 2026
- Built reporting pipelines for ESG data.

EDUCATION
Leiden University Leiden, Netherlands
MSc Computer Science Feb 2020 - Feb 2021
"""


def test_convert_to_pdf_uses_latex_renderer_for_resume(monkeypatch, tmp_path):
    resume_path = tmp_path / "tailored_resume.txt"
    output_path = tmp_path / "tailored_resume.pdf"
    resume_path.write_text(RESUME_TEXT, encoding="utf-8")
    captured = {}

    def fake_render_latex(latex, output):
        captured["latex"] = latex
        output.write_bytes(b"%PDF-1.4\n")
        return output

    monkeypatch.setenv("APPLYPILOT_RESUME_RENDERER", "latex")
    monkeypatch.setattr(pdf, "load_env", lambda: None)
    monkeypatch.setattr(
        pdf,
        "load_profile",
        lambda: {
            "personal": {
                "full_name": "Test Candidate",
                "city": "The Hague",
                "country": "Netherlands",
                "email": "candidate@example.com",
            },
            "work_authorization": {"work_permit_type": "EU citizen"},
            "resume_facts": {
                "preserved_companies": ["KPMG"],
                "employment_type_by_company": {"KPMG": "contractor"},
            },
        },
    )
    monkeypatch.setattr(pdf, "render_latex_pdf", fake_render_latex)

    result = pdf.convert_to_pdf(resume_path, output_path=output_path)

    assert result == output_path
    assert output_path.exists()
    assert "\\header{Experience}" in captured["latex"]
    assert "Senior Data Engineer, contractor" in captured["latex"]
    assert "EU citizen" in captured["latex"]


def test_convert_to_pdf_uses_moderncv_renderer_for_cover_letters(monkeypatch, tmp_path):
    cover_path = tmp_path / "job_CL.txt"
    output_path = tmp_path / "job_CL.pdf"
    cover_path.write_text(
        "Dear Hiring Manager,\n\nI built Python data pipelines for reporting teams.\n\nTest Candidate",
        encoding="utf-8",
    )
    captured = {}

    def fake_render_latex(latex, output):
        captured["latex"] = latex
        output.write_bytes(b"%PDF-1.4\n")
        return output

    monkeypatch.setattr(pdf, "load_env", lambda: None)
    monkeypatch.setattr(pdf, "render_latex_pdf", fake_render_latex)

    result = pdf.convert_to_pdf(
        cover_path,
        output_path=output_path,
        job={"title": "Data Engineer", "site": "ExampleCo"},
        profile={
            "personal": {
                "full_name": "Test Candidate",
                "city": "The Hague",
                "country": "Netherlands",
                "email": "candidate@example.com",
                "phone": "+31000000000",
            },
            "experience": {"target_role": "Senior Data Engineer"},
            "work_authorization": {"work_permit_type": "EU citizen"},
        },
    )

    assert result == output_path
    assert output_path.exists()
    assert "\\documentclass[11pt,a4paper,roman]{moderncv}" in captured["latex"]
    assert "\\moderncvcolor{green}" in captured["latex"]
    assert "\\recipient{Hiring Manager}{ExampleCo}" in captured["latex"]
    assert "\\makelettertitle" in captured["latex"]
    assert "I built Python data pipelines for reporting teams." in captured["latex"]
    assert captured["latex"].count("Dear Hiring Manager") == 1


def test_cover_renderer_can_be_disabled(monkeypatch, tmp_path):
    cover_path = tmp_path / "job_CL.txt"
    output_path = tmp_path / "job_CL.pdf"
    cover_path.write_text("Dear Hiring Manager,\n\nI built Python data pipelines.", encoding="utf-8")

    def fail_latex(*_args, **_kwargs):
        raise AssertionError("cover letters should use HTML when APPLYPILOT_COVER_RENDERER=html")

    def fake_render_pdf(_html, output):
        Path(output).write_bytes(b"%PDF-1.4\n")

    monkeypatch.setenv("APPLYPILOT_COVER_RENDERER", "html")
    monkeypatch.setattr(pdf, "load_env", lambda: None)
    monkeypatch.setattr(pdf, "render_latex_pdf", fail_latex)
    monkeypatch.setattr(pdf, "render_pdf", fake_render_pdf)

    result = pdf.convert_to_pdf(cover_path, output_path=output_path)

    assert result == output_path
    assert output_path.exists()


def test_build_latex_cover_letter_strips_text_greeting_and_signoff():
    latex = build_latex_cover_letter(
        """Dear Hiring Manager,

I built Python and SQL data platforms for reporting teams.

That work matches this role.

Test Candidate""",
        profile={"personal": {"full_name": "Test Candidate", "country": "Netherlands"}},
        job={"title": "Data Engineer | Cmotions", "site": "linkedin"},
    )

    assert "\\recipient{Hiring Manager}{Cmotions}" in latex
    assert "I built Python and SQL data platforms for reporting teams." in latex
    assert "That work matches this role." in latex
    assert latex.count("Dear Hiring Manager") == 1
    assert latex.count("\\textbf{Test Candidate}") == 1


def test_batch_convert_skips_manual_review_tailored_resumes(monkeypatch, tmp_path):
    conn = init_db(tmp_path / "applypilot.db")
    accepted_path = tmp_path / "accepted.txt"
    manual_path = tmp_path / "manual.txt"
    sidecar_path = tmp_path / "accepted_JOB.txt"
    accepted_path.write_text(RESUME_TEXT, encoding="utf-8")
    manual_path.write_text(RESUME_TEXT, encoding="utf-8")
    sidecar_path.write_text("job description", encoding="utf-8")
    conn.executemany(
        """
        INSERT INTO jobs (
            url, title, site, discovered_at, full_description, fit_score,
            tailored_resume_path, review_status
        )
        VALUES (?, ?, ?, ?, ?, ?, ?, ?)
        """,
        [
            (
                "https://example.com/accepted",
                "Data Engineer",
                "ExampleCo",
                "2026-05-25T18:00:00+00:00",
                "Accepted job.",
                8,
                str(accepted_path),
                None,
            ),
            (
                "https://example.com/manual",
                "Data Engineer",
                "ExampleCo",
                "2026-05-25T18:00:00+00:00",
                "Manual review job.",
                8,
                str(manual_path),
                "manual_review",
            ),
        ],
    )
    conn.commit()
    converted = []

    def fake_convert(path):
        converted.append(path)
        path.with_suffix(".pdf").write_bytes(b"%PDF-1.4\n")
        return path.with_suffix(".pdf")

    monkeypatch.setattr(pdf, "TAILORED_DIR", tmp_path)
    monkeypatch.setattr(database, "get_connection", lambda: conn)
    monkeypatch.setattr(pdf, "convert_to_pdf", fake_convert)

    assert pdf.batch_convert() == 1
    assert converted == [accepted_path]
    assert not manual_path.with_suffix(".pdf").exists()
    assert not sidecar_path.with_suffix(".pdf").exists()
    conn.close()
