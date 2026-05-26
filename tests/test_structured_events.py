import json
import sys
import types

from applypilot import events
from applypilot import pipeline
from applypilot.database import init_db
from applypilot.scoring import tailor


def _read_events(path):
    return [json.loads(line) for line in path.read_text(encoding="utf-8").splitlines()]


def _profile() -> dict:
    return {
        "personal": {
            "full_name": "Test Candidate",
            "email": "candidate@example.com",
            "phone": "+31000000000",
            "linkedin_url": "https://www.linkedin.com/in/test/",
        },
        "experience": {"education_level": "Master's"},
        "skills_boundary": {"languages": ["Python", "SQL"]},
        "resume_facts": {"preserved_school": "Leiden University"},
    }


def test_emit_event_writes_jsonl_and_redacts_secrets(monkeypatch, tmp_path):
    log_path = tmp_path / "events.jsonl"
    monkeypatch.setenv("APPLYPILOT_EVENT_LOG", str(log_path))
    events.start_run("test-run")

    record = events.emit_event(
        "unit_test_event",
        stage="tailor",
        status="failed",
        password="do-not-log",
        raw_excerpt="x" * 1100,
        nested={"api_key": "secret-key", "safe": "value"},
    )

    written = _read_events(log_path)

    assert record["run_id"] == "test-run"
    assert written == [record]
    assert written[0]["password"] == "[redacted]"
    assert written[0]["nested"]["api_key"] == "[redacted]"
    assert written[0]["nested"]["safe"] == "value"
    assert written[0]["raw_excerpt"].endswith("[truncated 100 chars]")


def test_tailoring_invalid_json_emits_structured_failure(monkeypatch, tmp_path):
    class FakeClient:
        def chat(self, *_args, **_kwargs):
            return "Here is a polished resume in prose, not JSON."

    log_path = tmp_path / "events.jsonl"
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

    monkeypatch.setenv("APPLYPILOT_EVENT_LOG", str(log_path))
    monkeypatch.setattr(tailor, "get_connection", lambda: conn)
    monkeypatch.setattr(tailor, "get_client", lambda: FakeClient())
    monkeypatch.setattr(tailor, "get_humanizer_prompt", lambda target: "")
    monkeypatch.setattr(tailor, "load_profile", _profile)
    monkeypatch.setattr(tailor, "RESUME_PATH", resume_path)
    monkeypatch.setattr(tailor, "TAILORED_DIR", tailored_dir)
    events.start_run("tailor-test")

    result = tailor.run_tailoring(limit=1, validation_mode="lenient")
    written = _read_events(log_path)
    failure = next(event for event in written if event["event"] == "tailor_job_failed")
    started_attempts = [event for event in written if event["event"] == "tailor_attempt_started"]
    invalid_attempts = [event for event in written if event["event"] == "tailor_attempt_invalid_json"]

    assert result["failed"] == 1
    assert len(started_attempts) == 4
    assert len(invalid_attempts) == 4
    assert started_attempts[0]["job_url"] == "https://example.com/job"
    assert started_attempts[0]["attempt"] == 1
    assert started_attempts[0]["max_attempts"] == 4
    assert failure["run_id"] == "tailor-test"
    assert failure["stage"] == "tailor"
    assert failure["status"] == "invalid_json"
    assert failure["job_url"] == "https://example.com/job"
    assert failure["parse_error_count"] == 4
    assert failure["first_parse_error"]["raw_excerpt"] == "Here is a polished resume in prose, not JSON."
    assert failure["report_path"].endswith("_REPORT.json")
    assert not (tailored_dir / "ExampleCo_Data_Engineer.txt").exists()
    conn.close()


def test_sequential_pipeline_emits_stage_error_events(monkeypatch, tmp_path):
    log_path = tmp_path / "events.jsonl"
    monkeypatch.setenv("APPLYPILOT_EVENT_LOG", str(log_path))
    events.start_run("pipeline-test")

    def fail_stage():
        raise RuntimeError("stage exploded")

    monkeypatch.setitem(pipeline._STAGE_RUNNERS, "score", fail_stage)

    result = pipeline._run_sequential(["score"], min_score=7)
    written = _read_events(log_path)

    assert result["errors"]["score"] == "error: stage exploded"
    assert any(event["event"] == "stage_started" and event["stage"] == "score" for event in written)
    assert any(event["event"] == "stage_failed" and event["error"] == "stage exploded" for event in written)
    assert any(
        event["event"] == "stage_finished" and event["stage"] == "score" and event["status"] == "error: stage exploded"
        for event in written
    )


def test_tailor_stage_reports_partial_when_some_jobs_fail(monkeypatch):
    fake_tailor = types.ModuleType("applypilot.scoring.tailor")
    fake_tailor.run_tailoring = lambda **_kwargs: {"approved": 14, "failed": 4, "errors": 0, "elapsed": 1.0}
    monkeypatch.setitem(sys.modules, "applypilot.scoring.tailor", fake_tailor)

    result = pipeline._run_tailor()

    assert result["status"] == "partial"
    assert result["failed"] == 4


def test_cover_stage_reports_partial_when_jobs_error(monkeypatch):
    fake_cover = types.ModuleType("applypilot.scoring.cover_letter")
    fake_cover.run_cover_letters = lambda **_kwargs: {"generated": 2, "errors": 1, "elapsed": 1.0}
    monkeypatch.setitem(sys.modules, "applypilot.scoring.cover_letter", fake_cover)

    result = pipeline._run_cover()

    assert result["status"] == "partial"
    assert result["errors"] == 1
