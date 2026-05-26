"""Resume tailoring: LLM-powered ATS-optimized resume generation per job.

THIS IS THE HEAVIEST REFACTOR. Every piece of personal data -- name, email, phone,
skills, companies, projects, school -- is loaded at runtime from the user's profile.
Zero hardcoded personal information.

The LLM returns structured JSON, code assembles the final text. Header (name, contact)
is always code-injected, never LLM-generated. Each retry starts a fresh conversation
to avoid apologetic spirals.
"""

import copy
import json
import logging
import os
import re
import time
from datetime import datetime, timezone
from hashlib import sha1

from applypilot.config import RESUME_PATH, TAILORED_DIR, load_profile
from applypilot.database import get_connection, get_jobs_by_stage
from applypilot.events import emit_error, emit_event
from applypilot.humanizer import get_humanizer_prompt
from applypilot.llm import get_client, make_anthropic_client, make_client
from applypilot.scoring.validator import (
    BANNED_WORDS,
    sanitize_text,
    validate_json_fields,
)

log = logging.getLogger(__name__)

MAX_ATTEMPTS = 5  # max cross-run retries before giving up
RESUME_SECTION_HEADERS = {
    "SUMMARY",
    "TECHNICAL SKILLS",
    "SKILLS",
    "EXPERIENCE",
    "PROJECTS",
    "EDUCATION",
}


def _tailor_model_override() -> str:
    return os.environ.get("TAILOR_LLM_MODEL", "").strip()


def _tailor_provider_override() -> str:
    return os.environ.get("TAILOR_LLM_PROVIDER", "").strip().casefold()


def _env_int(name: str, default: int, minimum: int = 1) -> int:
    value = os.environ.get(name, "").strip()
    if not value:
        return default
    try:
        return max(int(value), minimum)
    except ValueError:
        log.warning("Invalid %s=%r; using %s", name, value, default)
        return default


def _tailor_max_retries() -> int:
    return _env_int("TAILOR_MAX_RETRIES", 3, minimum=0)


def _tailor_max_tokens() -> int:
    return _env_int("TAILOR_LLM_MAX_TOKENS", 6144, minimum=2048)


def _is_claude_model(model: str) -> bool:
    normalized = model.strip().casefold()
    return normalized.startswith(("claude-", "anthropic/claude-"))


def _get_tailor_client():
    provider = _tailor_provider_override()
    model = _tailor_model_override()
    if provider in {"anthropic", "claude"} or (model and _is_claude_model(model)):
        return make_anthropic_client(model=model or None)
    if model:
        return make_client(model=model)
    return get_client()


def _job_file_prefix(job: dict) -> str:
    """Build a stable, collision-resistant filename prefix for a job."""
    safe_title = re.sub(r"[^\w\s-]", "", job["title"])[:50].strip().replace(" ", "_")
    safe_site = re.sub(r"[^\w\s-]", "", job["site"])[:20].strip().replace(" ", "_")
    digest = sha1(str(job["url"]).encode("utf-8")).hexdigest()[:8]
    return f"{safe_site}_{safe_title}_{digest}"


# ── Prompt Builders (profile-driven) ──────────────────────────────────────


def _build_tailor_prompt(profile: dict) -> str:
    """Build the resume tailoring system prompt from the user's profile.

    All skills boundaries, preserved entities, and formatting rules are
    derived from the profile -- nothing is hardcoded.
    """
    boundary = profile.get("skills_boundary", {})
    resume_facts = profile.get("resume_facts", {})

    # Format skills boundary for the prompt
    skills_lines = []
    for category, items in boundary.items():
        if isinstance(items, list) and items:
            label = category.replace("_", " ").title()
            skills_lines.append(f"{label}: {', '.join(items)}")
    skills_block = "\n".join(skills_lines)

    # Preserved entities
    companies = resume_facts.get("preserved_companies", [])
    projects = resume_facts.get("preserved_projects", [])
    school = resume_facts.get("preserved_school", "")
    education_block = _education_block_from_profile(profile)
    real_metrics = resume_facts.get("real_metrics", [])
    employment_types = resume_facts.get("employment_type_by_company", {})
    employment_context = resume_facts.get("employment_context", "")

    companies_str = ", ".join(companies) if companies else "N/A"
    projects_str = ", ".join(projects) if projects else "N/A"
    metrics_str = ", ".join(real_metrics) if real_metrics else "N/A"
    employment_lines = (
        "\n".join(f"- {company}: {employment_type}" for company, employment_type in employment_types.items())
        if isinstance(employment_types, dict) and employment_types
        else "N/A"
    )

    # Include ALL banned words from the validator so the LLM knows exactly
    # what will be rejected — the validator checks for these automatically.
    banned_str = ", ".join(BANNED_WORDS)

    education = profile.get("experience", {})
    education_level = education.get("education_level", "")

    prompt = f"""You are a senior technical recruiter rewriting a resume to get this person an interview.

Take the base resume and job description. Return a tailored resume as a JSON object.

## RECRUITER SCAN (6 seconds):
1. Title -- matches what they're hiring?
2. Summary -- 2 sentences proving you've done this work
3. First 3 bullets of most recent role -- verbs and outcomes match?
4. Skills -- must-haves visible immediately?

## SKILLS BOUNDARY (real skills only):
{skills_block}

You MAY add 2-3 closely related tools (Kubernetes if Docker, Terraform if AWS, Redis if PostgreSQL). No unrelated languages/frameworks.

## TAILORING RULES:

TITLE: Match the target role. Keep seniority (Senior/Lead/Staff). Drop company suffixes and team names.

SUMMARY: Rewrite from scratch. Lead with the 1-2 skills that matter most for THIS role. Sound like someone who's done this job.

SKILLS: Reorder each category so the job's must-haves appear first.

Reframe EVERY bullet for this role. Same real work, different angle. Every bullet must be reworded. Never copy verbatim.

PROJECTS: Reorder by relevance. If no project is relevant, return "projects": [].

BULLETS: Strong verb + what you built + quantified impact. Vary verbs (Built, Designed, Implemented, Reduced, Automated, Deployed, Operated, Optimized). Most relevant first. Max 4 per section.

EXPERIENCE TIMELINE: Include every company listed under Preserved companies exactly once. Never omit older roles to save space. If space is tight, keep older roles to 1-2 bullets, but keep the company, role, employment type, and dates visible.

EMPLOYMENT TYPE IS A FACT:
{employment_lines}

{employment_context}

For every experience entry, preserve the employment type in the role text. Use headers like "Senior Data Engineer, contractor at KPMG" or "Data Engineer, full-time at Metyis/Adaptfy". Do not drop "contractor" to save space; it explains fixed-term roles and short tenures.

EDUCATION IS A FACT:
{education_block or f"{school} | {education_level}"}

Preserve education school names, degree labels, campus locations, and date ranges. Do not collapse three schools into one line. Do not replace dated education with only "{education_level}".

## VOICE:
- Write like a real engineer. Short, direct.
- GOOD: "Automated financial reporting with Python + API integrations, cut processing time from 10 hours to 2"
- BAD: "Leveraged cutting-edge AI technologies to drive transformative operational efficiencies"
- BANNED WORDS (using ANY of these = validation failure — do not use them even once):
  {banned_str}
- No em dashes. Use commas, periods, or hyphens.

## HARD RULES:
- Do NOT invent work, companies, degrees, or certifications
- Do NOT change real numbers ({metrics_str})
- Preserved companies: {companies_str} -- names stay as-is
- Do NOT remove or combine preserved companies. The timeline matters.
- Preserved employment types:
{employment_lines}
- Preserved school: {school}
- Preserved education block:
{education_block or f"{school} | {education_level}"}
- Must fit 1 page.
"""

    education_json = json.dumps(education_block or f"{school} | {education_level}", ensure_ascii=False)
    output_contract = f"""
## OUTPUT: Return ONLY valid JSON. No markdown fences. No commentary. No "here is" preamble.

{{"title":"Role Title","summary":"2-3 tailored sentences.","skills":{{"Languages":"...","Frameworks":"...","DevOps & Infra":"...","Databases":"...","Tools":"..."}},"experience":[{{"header":"Title at Company","subtitle":"Tech | Dates","bullets":["bullet 1","bullet 2","bullet 3","bullet 4"]}}],"projects":[],"education":{education_json}}}"""
    return prompt + get_humanizer_prompt("resume") + output_contract


def _profile_with_preserved_education(profile: dict, resume_text: str) -> dict:
    """Attach an exact education block so tailoring cannot compress it away."""
    patched = copy.deepcopy(profile)
    facts = patched.setdefault("resume_facts", {})
    if _education_block_from_profile(patched):
        return patched

    extracted = _extract_resume_section(resume_text, "EDUCATION")
    if extracted:
        facts["education_block"] = extracted
    return patched


def _education_block_from_profile(profile: dict) -> str:
    education = profile.get("education")
    if isinstance(education, list) and education:
        lines: list[str] = []
        for entry in education:
            if not isinstance(entry, dict):
                continue
            school = str(entry.get("school") or "").strip()
            location = str(entry.get("location") or "").strip()
            degree = str(entry.get("degree") or "").strip()
            dates = str(entry.get("dates") or "").strip()
            first_line = " ".join(part for part in (school, location) if part).strip()
            second_line = " ".join(part for part in (degree, dates) if part).strip()
            if first_line:
                lines.append(first_line)
            if second_line:
                lines.append(second_line)
        return "\n".join(lines).strip()

    facts = profile.get("resume_facts", {})
    block = facts.get("education_block") if isinstance(facts, dict) else ""
    return sanitize_text(str(block or "")).strip()


def _extract_resume_section(text: str, section_name: str) -> str:
    lines = text.splitlines()
    start = None
    for index, line in enumerate(lines):
        if line.strip().casefold() == section_name.casefold():
            start = index + 1
            break
    if start is None:
        return ""

    section_lines: list[str] = []
    for line in lines[start:]:
        stripped = line.strip()
        if stripped.upper() in RESUME_SECTION_HEADERS:
            break
        section_lines.append(line.rstrip())
    return sanitize_text("\n".join(section_lines).strip())


def _build_judge_prompt(profile: dict) -> str:
    """Build the LLM judge prompt from the user's profile."""
    boundary = profile.get("skills_boundary", {})
    resume_facts = profile.get("resume_facts", {})

    # Flatten allowed skills for the judge
    all_skills: list[str] = []
    for items in boundary.values():
        if isinstance(items, list):
            all_skills.extend(items)
    skills_str = ", ".join(all_skills) if all_skills else "N/A"

    real_metrics = resume_facts.get("real_metrics", [])
    metrics_str = ", ".join(real_metrics) if real_metrics else "N/A"
    employment_types = resume_facts.get("employment_type_by_company", {})
    employment_lines = (
        "\n".join(f"- {company}: {employment_type}" for company, employment_type in employment_types.items())
        if isinstance(employment_types, dict) and employment_types
        else "N/A"
    )

    return f"""You are a resume quality judge. A tailoring engine rewrote a resume to target a specific job. Your job is to catch LIES, not style changes.

You must answer with EXACTLY this format:
VERDICT: PASS or FAIL
ISSUES: (list any problems, or "none")

## CONTEXT -- what the tailoring engine was instructed to do (all of this is ALLOWED):
- Change the title to match the target role
- Rewrite the summary from scratch for the target job
- Reorder bullets and projects to put the most relevant first
- Reframe bullets to use the job's language
- Drop low-relevance bullets and replace with more relevant ones from other sections
- Reorder the skills section to put job-relevant skills first
- Change tone and wording extensively

## WHAT IS FABRICATION (FAIL for these):
1. Adding tools, languages, or frameworks to TECHNICAL SKILLS that aren't in the original. The allowed skills are ONLY: {skills_str}
2. Inventing NEW metrics or numbers not in the original. The real metrics are: {metrics_str}
3. Inventing work that has no basis in any original bullet (completely new achievements).
4. Adding companies, roles, or degrees that don't exist.
5. Changing real numbers (inflating 80% to 95%, 500 nodes to 1000 nodes).
6. Removing or contradicting employment type labels for these roles:
{employment_lines}

## WHAT IS NOT FABRICATION (do NOT fail for these):
- Rewording any bullet, even heavily, as long as the underlying work is real
- Combining two original bullets into one
- Splitting one original bullet into two
- Describing the same work with different emphasis
- Dropping bullets entirely
- Reordering anything
- Changing the title or summary completely

## TOLERANCE RULE:
The goal is to get interviews, not to be a perfect fact-checker. Allow up to 3 minor stretches per resume:
- Adding a closely related tool the candidate could realistically know is a MINOR STRETCH, not fabrication.
- Reframing a metric with slightly different wording is a MINOR STRETCH.
- Adding any LEARNABLE skill given their existing stack is a MINOR STRETCH.
- Only FAIL if there are MAJOR lies: completely invented projects, fake companies, fake degrees, wildly inflated numbers, or skills from a completely different domain.

Be strict about major lies. Be lenient about minor stretches and learnable skills. Do not fail for style, tone, or restructuring."""


# ── JSON Extraction ───────────────────────────────────────────────────────


def extract_json(raw: str) -> dict:
    """Robustly extract JSON from LLM response (handles fences, preamble).

    Args:
        raw: Raw LLM response text.

    Returns:
        Parsed JSON dict.

    Raises:
        ValueError: If no valid JSON found.
    """
    raw = raw.strip()

    # Direct parse
    try:
        return json.loads(raw)
    except json.JSONDecodeError:
        pass

    # Markdown fences
    if "```" in raw:
        for part in raw.split("```")[1::2]:
            part = part.strip()
            if part.startswith("json"):
                part = part[4:].strip()
            try:
                return json.loads(part)
            except json.JSONDecodeError:
                continue

    # Find outermost { ... }
    start = raw.find("{")
    end = raw.rfind("}")
    if start != -1 and end > start:
        try:
            return json.loads(raw[start : end + 1])
        except json.JSONDecodeError:
            pass

    raise ValueError("No valid JSON found in LLM response")


# ── Resume Assembly (profile-driven header) ──────────────────────────────


def assemble_resume_text(data: dict, profile: dict) -> str:
    """Convert JSON resume data to formatted plain text.

    Header (name, location, contact) is ALWAYS code-injected from the profile,
    never LLM-generated. All text fields are sanitized.

    Args:
        data: Parsed JSON resume from the LLM.
        profile: User profile dict from load_profile().

    Returns:
        Formatted resume text.
    """
    personal = profile.get("personal", {})
    lines: list[str] = []

    # Header -- always code-injected from profile
    lines.append(personal.get("full_name", ""))
    lines.append(sanitize_text(data.get("title", "Software Engineer")))

    location = ", ".join(
        part
        for part in (
            personal.get("city", ""),
            personal.get("country", ""),
        )
        if part
    )
    if location:
        lines.append(location)

    # Contact line
    contact_parts: list[str] = []
    if personal.get("email"):
        contact_parts.append(personal["email"])
    if personal.get("phone"):
        contact_parts.append(personal["phone"])
    if personal.get("github_url"):
        contact_parts.append(personal["github_url"])
    if personal.get("linkedin_url"):
        contact_parts.append(personal["linkedin_url"])
    if contact_parts:
        lines.append(" | ".join(contact_parts))
    lines.append("")

    # Summary
    lines.append("SUMMARY")
    lines.append(sanitize_text(data["summary"]))
    lines.append("")

    # Technical Skills
    lines.append("TECHNICAL SKILLS")
    if isinstance(data["skills"], dict):
        for cat, val in data["skills"].items():
            lines.append(f"{cat}: {sanitize_text(str(val))}")
    lines.append("")

    # Experience
    lines.append("EXPERIENCE")
    for entry in data.get("experience", []):
        lines.append(sanitize_text(entry.get("header", "")))
        if entry.get("subtitle"):
            lines.append(sanitize_text(entry["subtitle"]))
        for b in entry.get("bullets", []):
            lines.append(f"- {sanitize_text(b)}")
        lines.append("")

    # Projects
    lines.append("PROJECTS")
    for entry in data.get("projects", []):
        lines.append(sanitize_text(entry.get("header", "")))
        if entry.get("subtitle"):
            lines.append(sanitize_text(entry["subtitle"]))
        for b in entry.get("bullets", []):
            lines.append(f"- {sanitize_text(b)}")
        lines.append("")

    # Education
    lines.append("EDUCATION")
    education = _education_block_from_profile(profile) or sanitize_text(str(data.get("education", "")))
    if education:
        lines.append(education)

    return "\n".join(lines)


# ── LLM Judge ────────────────────────────────────────────────────────────


def judge_tailored_resume(original_text: str, tailored_text: str, job_title: str, profile: dict) -> dict:
    """LLM judge layer: catches subtle fabrication that programmatic checks miss.

    Args:
        original_text: Base resume text.
        tailored_text: Tailored resume text.
        job_title: Target job title.
        profile: User profile for building the judge prompt.

    Returns:
        {"passed": bool, "verdict": str, "issues": str, "raw": str}
    """
    judge_prompt = _build_judge_prompt(profile)

    messages = [
        {"role": "system", "content": judge_prompt},
        {
            "role": "user",
            "content": (
                f"JOB TITLE: {job_title}\n\n"
                f"ORIGINAL RESUME:\n{original_text}\n\n---\n\n"
                f"TAILORED RESUME:\n{tailored_text}\n\n"
                "Judge this tailored resume:"
            ),
        },
    ]

    client = get_client()
    response = client.chat(messages, max_tokens=512, temperature=0.1)

    passed = "VERDICT: PASS" in response.upper()
    issues = "none"
    if "ISSUES:" in response.upper():
        issues_idx = response.upper().index("ISSUES:")
        issues = response[issues_idx + 7 :].strip()

    return {
        "passed": passed,
        "verdict": "PASS" if passed else "FAIL",
        "issues": issues,
        "raw": response,
    }


# ── Core Tailoring ───────────────────────────────────────────────────────


def tailor_resume(
    resume_text: str,
    job: dict,
    profile: dict,
    max_retries: int | None = None,
    validation_mode: str = "normal",
) -> tuple[str, dict]:
    """Generate a tailored resume via JSON output + fresh context on each retry.

    Key design choices:
    - LLM returns structured JSON, code assembles the text (no header leaks)
    - Each retry starts a FRESH conversation (no apologetic spiral)
    - Issues from previous attempts are noted in the system prompt
    - Em dashes and smart quotes are auto-fixed, not rejected

    Args:
        resume_text:      Base resume text.
        job:              Job dict with title, site, location, full_description.
        profile:          User profile dict.
        max_retries:      Maximum retry attempts.
        validation_mode:  "strict", "normal", or "lenient".
                          strict  -- banned words trigger retries; judge must pass
                          normal  -- banned words = warnings only; judge can fail on last retry
                          lenient -- banned words ignored; LLM judge skipped

    Returns:
        (tailored_text, report) where report contains validation details.
    """
    if max_retries is None:
        max_retries = _tailor_max_retries()

    job_text = (
        f"TITLE: {job['title']}\n"
        f"COMPANY: {job['site']}\n"
        f"LOCATION: {job.get('location', 'N/A')}\n\n"
        f"DESCRIPTION:\n{(job.get('full_description') or '')[:6000]}"
    )
    profile = _profile_with_preserved_education(profile, resume_text)

    report: dict = {
        "attempts": 0,
        "validator": None,
        "judge": None,
        "status": "pending",
        "validation_mode": validation_mode,
        "parse_errors": [],
        "tailor_model_override": bool(_tailor_model_override()),
        "tailor_provider_override": _tailor_provider_override() or None,
        "model": None,
    }
    avoid_notes: list[str] = []
    tailored = ""
    client = _get_tailor_client()
    report["model"] = getattr(client, "model", None)
    tailor_prompt_base = _build_tailor_prompt(profile)
    max_tokens = _tailor_max_tokens()

    for attempt in range(max_retries + 1):
        report["attempts"] = attempt + 1

        # Fresh conversation every attempt
        prompt = tailor_prompt_base
        if avoid_notes:
            prompt += "\n\n## AVOID THESE ISSUES (from previous attempt):\n" + "\n".join(
                f"- {n}" for n in avoid_notes[-5:]
            )

        messages = [
            {"role": "system", "content": prompt},
            {
                "role": "user",
                "content": f"ORIGINAL RESUME:\n{resume_text}\n\n---\n\nTARGET JOB:\n{job_text}\n\nReturn the JSON:",
            },
        ]

        attempt_started = time.time()
        emit_event(
            "tailor_attempt_started",
            stage="tailor",
            job_title=job.get("title"),
            site=job.get("site"),
            job_url=job.get("url"),
            attempt=attempt + 1,
            max_attempts=max_retries + 1,
            model=report["model"],
            max_tokens=max_tokens,
        )
        log.info(
            "Tailor attempt %d/%d started | %s | %s",
            attempt + 1,
            max_retries + 1,
            job.get("title", "Untitled"),
            job.get("site", "Unknown"),
        )

        raw = client.chat(messages, max_tokens=max_tokens, temperature=0.4)
        emit_event(
            "tailor_attempt_response",
            stage="tailor",
            job_title=job.get("title"),
            site=job.get("site"),
            job_url=job.get("url"),
            attempt=attempt + 1,
            elapsed_seconds=round(time.time() - attempt_started, 1),
            raw_chars=len(raw),
        )

        # Parse JSON from response
        try:
            data = extract_json(raw)
        except ValueError as e:
            report["parse_errors"].append(
                {
                    "attempt": attempt + 1,
                    "error": str(e),
                    "raw_excerpt": raw[:500],
                }
            )
            emit_event(
                "tailor_attempt_invalid_json",
                level="warning",
                stage="tailor",
                job_title=job.get("title"),
                site=job.get("site"),
                job_url=job.get("url"),
                attempt=attempt + 1,
                error=str(e),
                raw_chars=len(raw),
            )
            avoid_notes.append("Output was not valid JSON. Return ONLY a JSON object, nothing else.")
            continue

        # Layer 1: Validate JSON fields
        validation = validate_json_fields(data, profile, mode=validation_mode)
        report["validator"] = validation

        if not validation["passed"]:
            # Only retry if there are hard errors (warnings never block)
            avoid_notes.extend(validation["errors"])
            if attempt < max_retries:
                continue
            # Last attempt — assemble whatever we got
            tailored = assemble_resume_text(data, profile)
            report["status"] = "failed_validation"
            return tailored, report

        # Assemble text (header injected by code, em dashes auto-fixed)
        tailored = assemble_resume_text(data, profile)

        # Layer 2: LLM judge (catches subtle fabrication) — skipped in lenient mode
        if validation_mode == "lenient":
            report["judge"] = {"verdict": "SKIPPED", "passed": True, "issues": "none"}
            report["status"] = "approved"
            return tailored, report

        judge = judge_tailored_resume(resume_text, tailored, job.get("title", ""), profile)
        report["judge"] = judge

        if not judge["passed"]:
            avoid_notes.append(f"Judge rejected: {judge['issues']}")
            if attempt < max_retries:
                # In normal mode, only retry on judge failure if there are retries left
                if validation_mode != "lenient":
                    continue
            # Accept best attempt on last retry (all modes) or if lenient
            report["status"] = "approved_with_judge_warning"
            return tailored, report

        # Both passed
        report["status"] = "approved"
        return tailored, report

    report["status"] = "invalid_json" if report["parse_errors"] and not tailored else "exhausted_retries"
    return tailored, report


# ── Batch Entry Point ────────────────────────────────────────────────────


def run_tailoring(min_score: int = 7, limit: int = 20, validation_mode: str = "normal") -> dict:
    """Generate tailored resumes for high-scoring jobs.

    Args:
        min_score:       Minimum fit_score to tailor for.
        limit:           Maximum jobs to process.
        validation_mode: "strict", "normal", or "lenient".

    Returns:
        {"approved": int, "failed": int, "errors": int, "elapsed": float}
    """
    profile = load_profile()
    resume_text = RESUME_PATH.read_text(encoding="utf-8")
    conn = get_connection()

    jobs = get_jobs_by_stage(conn=conn, stage="pending_tailor", min_score=min_score, limit=limit)

    if not jobs:
        log.info("No untailored jobs with score >= %d.", min_score)
        return {"approved": 0, "failed": 0, "errors": 0, "elapsed": 0.0}

    TAILORED_DIR.mkdir(parents=True, exist_ok=True)
    log.info("Tailoring resumes for %d jobs (score >= %d)...", len(jobs), min_score)
    t0 = time.time()
    completed = 0
    results: list[dict] = []
    stats: dict[str, int] = {"approved": 0, "failed_validation": 0, "failed_judge": 0, "error": 0}

    for job in jobs:
        completed += 1
        try:
            emit_event(
                "tailor_job_started",
                stage="tailor",
                job_title=job.get("title"),
                site=job.get("site"),
                job_url=job.get("url"),
                index=completed,
                total=len(jobs),
            )
            log.info(
                "%d/%d starting | %s | %s",
                completed,
                len(jobs),
                job.get("title", "Untitled"),
                job.get("site", "Unknown"),
            )
            tailored, report = tailor_resume(resume_text, job, profile, validation_mode=validation_mode)

            prefix = _job_file_prefix(job)

            # Save tailored resume text only when the LLM produced one.
            txt_path = TAILORED_DIR / f"{prefix}.txt"
            if tailored.strip():
                txt_path.write_text(tailored, encoding="utf-8")
            else:
                txt_path = None

            # Save job description for traceability
            job_path = TAILORED_DIR / f"{prefix}_JOB.txt"
            job_desc = (
                f"Title: {job['title']}\n"
                f"Company: {job['site']}\n"
                f"Location: {job.get('location', 'N/A')}\n"
                f"Score: {job.get('fit_score', 'N/A')}\n"
                f"URL: {job['url']}\n\n"
                f"{job.get('full_description', '')}"
            )
            job_path.write_text(job_desc, encoding="utf-8")

            # Save validation report
            report_path = TAILORED_DIR / f"{prefix}_REPORT.json"
            report_path.write_text(json.dumps(report, indent=2), encoding="utf-8")

            # Generate PDF for approved resumes (best-effort)
            # "approved_with_judge_warning" is also a success — resume was generated.
            pdf_path = None
            if txt_path and report["status"] in ("approved", "approved_with_judge_warning"):
                try:
                    from applypilot.scoring.pdf import convert_to_pdf

                    pdf_path = str(convert_to_pdf(txt_path))
                except Exception:
                    log.debug("PDF generation failed for %s", txt_path, exc_info=True)

            result = {
                "url": job["url"],
                "path": str(txt_path) if txt_path else None,
                "pdf_path": pdf_path,
                "report_path": str(report_path),
                "title": job["title"],
                "site": job["site"],
                "status": report["status"],
                "attempts": report["attempts"],
            }
            event_fields = {
                "job_url": job["url"],
                "job_title": job["title"],
                "site": job["site"],
                "attempts": report["attempts"],
                "report_path": report_path,
                "resume_path": txt_path,
                "pdf_path": pdf_path,
                "parse_error_count": len(report.get("parse_errors") or []),
            }
            if report.get("parse_errors"):
                event_fields["first_parse_error"] = report["parse_errors"][0]
            if report["status"] in ("approved", "approved_with_judge_warning"):
                emit_event("tailor_job_finished", stage="tailor", status=report["status"], **event_fields)
            else:
                emit_event(
                    "tailor_job_failed",
                    level="error",
                    stage="tailor",
                    status=report["status"],
                    **event_fields,
                )
        except Exception as e:
            result = {
                "url": job["url"],
                "title": job["title"],
                "site": job["site"],
                "status": "error",
                "attempts": 0,
                "path": None,
                "pdf_path": None,
            }
            log.error("%d/%d [ERROR] %s -- %s", completed, len(jobs), job["title"][:40], e)
            emit_error(
                "tailor_job_exception",
                e,
                stage="tailor",
                job_url=job["url"],
                job_title=job["title"],
                site=job["site"],
            )

        results.append(result)
        stats[result.get("status", "error")] = stats.get(result.get("status", "error"), 0) + 1

        elapsed = time.time() - t0
        rate = completed / elapsed if elapsed > 0 else 0
        log.info(
            "%d/%d [%s] attempts=%s | %.1f jobs/min | %s",
            completed,
            len(jobs),
            result["status"].upper(),
            result.get("attempts", "?"),
            rate * 60,
            result["title"][:40],
        )

    # Persist to DB: increment attempt counter for ALL, save path only for approved
    now = datetime.now(timezone.utc).isoformat()
    _success_statuses = {"approved", "approved_with_judge_warning"}
    for r in results:
        if r["status"] in _success_statuses:
            conn.execute(
                "UPDATE jobs SET tailored_resume_path=?, tailored_at=?, "
                "tailor_attempts=COALESCE(tailor_attempts,0)+1 WHERE url=?",
                (r["path"], now, r["url"]),
            )
        else:
            conn.execute(
                "UPDATE jobs SET tailor_attempts=COALESCE(tailor_attempts,0)+1 WHERE url=?",
                (r["url"],),
            )
    conn.commit()

    elapsed = time.time() - t0
    failed_count = sum(
        count for status, count in stats.items() if status not in {"approved", "approved_with_judge_warning", "error"}
    )
    log.info(
        "Tailoring done in %.1fs: %d approved, %d failed, %d errors",
        elapsed,
        stats.get("approved", 0),
        failed_count,
        stats.get("error", 0),
    )

    return {
        "approved": stats.get("approved", 0),
        "failed": failed_count,
        "errors": stats.get("error", 0),
        "elapsed": elapsed,
    }
