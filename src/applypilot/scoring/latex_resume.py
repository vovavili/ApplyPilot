"""LaTeX resume renderer using the user's compact CV style."""

from __future__ import annotations

import os
import re
import shutil
import subprocess
from pathlib import Path


DEFAULT_PREAMBLE = r"""
\documentclass[a4paper]{article}
\usepackage{fullpage}
\usepackage{amsmath}
\usepackage{amssymb}
\usepackage{textcomp}
\usepackage[utf8]{inputenc}
\usepackage[T1]{fontenc}
\usepackage{tabularx}
\usepackage{array}
\textheight=10in
\pagestyle{empty}
\raggedright
\usepackage[left=0.8in,right=0.8in,bottom=0.8in,top=0.8in]{geometry}

\def\bull{\vrule height 0.8ex width .7ex depth -.1ex }

\newcommand{\lineunder} {
	\vspace*{-8pt} \\
	\hspace*{-18pt} \hrulefill \\
}

\newcommand{\header} [1] {
	{\hspace*{-18pt}\vspace*{6pt} \textsc{#1}}
	\vspace*{-6pt} \lineunder
}
""".strip()

MONTH_PATTERN = re.compile(
    r"\b(?:Jan|Feb|Mar|Apr|May|Jun|Jul|Aug|Sep|Sept|Oct|Nov|Dec|"
    r"January|February|March|April|June|July|August|September|October|November|December)\b",
    re.IGNORECASE,
)


def build_latex_resume(resume: dict, profile: dict | None = None) -> str:
    """Build a one-page-ish LaTeX resume from ApplyPilot's parsed resume."""
    preamble = _load_preamble()
    body_sections = [
        _build_contact_header(resume, profile or {}),
        _build_summary(resume),
        _build_skills(resume),
        _build_entries("Experience", resume["sections"].get("EXPERIENCE", ""), profile or {}),
        _build_entries("Projects", resume["sections"].get("PROJECTS", ""), profile or {}),
        _build_education(resume["sections"].get("EDUCATION", "")),
    ]
    body = "\n\n".join(section for section in body_sections if section.strip())

    return f"""{preamble}

\\begin{{document}}
\t\\vspace*{{-40pt}}
{body}

\\end{{document}}
"""


def render_latex_pdf(latex: str, output_path: Path) -> Path:
    """Write a sibling .tex file and compile it to the requested PDF path."""
    output_path = Path(output_path).resolve()
    output_path.parent.mkdir(parents=True, exist_ok=True)
    tex_path = output_path.with_suffix(".tex")
    tex_path.write_text(latex, encoding="utf-8")

    pdflatex = _find_pdflatex()
    if not pdflatex:
        raise RuntimeError("pdflatex was not found. Install MiKTeX/TeX Live or put pdflatex on PATH.")

    result = subprocess.run(
        [
            pdflatex,
            "-interaction=nonstopmode",
            "-halt-on-error",
            "-output-directory",
            str(output_path.parent),
            str(tex_path),
        ],
        check=False,
        capture_output=True,
        text=True,
        encoding="utf-8",
        errors="replace",
    )
    if result.returncode != 0:
        log_tail = "\n".join((result.stdout + result.stderr).splitlines()[-40:])
        raise RuntimeError(f"pdflatex failed for {tex_path}:\n{log_tail}")

    if not output_path.exists():
        raise RuntimeError(f"pdflatex finished but did not create {output_path}")
    return output_path


def _load_preamble() -> str:
    template_path = os.environ.get("APPLYPILOT_LATEX_CV_TEMPLATE_PATH", "").strip()
    if not template_path:
        return DEFAULT_PREAMBLE

    path = Path(template_path).expanduser()
    if not path.exists():
        return DEFAULT_PREAMBLE

    template = path.read_text(encoding="utf-8")
    marker = r"\begin{document}"
    if marker not in template:
        return DEFAULT_PREAMBLE
    return template.split(marker, 1)[0].strip()


def _build_contact_header(resume: dict, profile: dict) -> str:
    personal = profile.get("personal", {})
    name = personal.get("full_name") or resume.get("name", "")

    contact_parts = _contact_parts(resume, profile)
    contact_line = r" $\cdot$ ".join(_latex_escape(part) for part in contact_parts if part)

    title = _clean_resume_title(resume.get("title", ""))
    title_line = f"\n\t\t{{\\large {_latex_escape(title)}}}\\\\" if title else ""

    return f"""
\t\\begin{{center}}
\t\t{{\\Huge \\scshape {{{_latex_escape(name)}}}}}\\\\{title_line}
\t\t{contact_line}\\\\
\t\\end{{center}}
""".rstrip()


def _contact_parts(resume: dict, profile: dict) -> list[str]:
    personal = profile.get("personal", {})
    work_auth = profile.get("work_authorization", {})

    city = personal.get("city", "")
    country = personal.get("country", "")
    location = ", ".join(part for part in (city, country) if part)
    if not location:
        location = resume.get("location", "")

    parts = [location]
    if personal.get("email"):
        parts.append(personal["email"])
    if personal.get("phone"):
        parts.append(personal["phone"])
    if work_auth.get("work_permit_type"):
        parts.append(work_auth["work_permit_type"])
    if personal.get("linkedin_url"):
        parts.append(personal["linkedin_url"])
    elif resume.get("contact"):
        parts.extend(part.strip() for part in resume["contact"].split("|") if part.strip())
    return parts


def _clean_resume_title(title: str) -> str:
    if "@" in title or "http" in title.lower() or "⋅" in title or "|" in title:
        return ""
    return title


def _build_summary(resume: dict) -> str:
    summary = resume["sections"].get("SUMMARY", "").strip()
    if not summary:
        return ""
    return f"""
\t\\header{{Summary}}
\t{_latex_escape(summary)}
\t\\vspace{{2mm}}
""".rstrip()


def _build_skills(resume: dict) -> str:
    skills = _parse_skills(resume["sections"].get("TECHNICAL SKILLS", "") or resume["sections"].get("SKILLS", ""))
    if not skills:
        return ""

    rows = "\n".join(
        f"\t\t{_latex_escape(_ensure_colon(category)):<24} & {_latex_escape(value)} \\\\"
        for category, value in skills
        if category and value
    )
    return f"""
\t\\header{{Skills}}
\t\\begin{{tabularx}}{{\\textwidth}}{{@{{}} >{{\\bfseries}}l @{{\\hspace{{6ex}}}} X }}
{rows}
\t\\end{{tabularx}}
\t\\vspace{{2mm}}
""".rstrip()


def _build_entries(section_name: str, text: str, profile: dict) -> str:
    entries = _parse_entries(text)
    if not entries:
        return ""

    rendered = []
    for entry in entries:
        company, location, role, dates = _split_work_entry(entry, profile)
        location_part = f" \\hfill {_latex_escape(location)}" if location else ""
        dates_part = f" \\hfill {_latex_escape(dates)}" if dates else ""
        bullets = "\n".join(f"\t\t\\item {_latex_escape(bullet)}" for bullet in entry["bullets"])
        itemize = (
            f"""
\t\\vspace{{-1mm}}
\t\\begin{{itemize}} \\itemsep 1pt
{bullets}
\t\\end{{itemize}}""".rstrip()
            if bullets
            else ""
        )
        subtitle = f"\n\t\\textit{{{_latex_escape(role)}}}{dates_part}\\\\" if role or dates else ""
        rendered.append(
            f"""
\t\\textbf{{{_latex_escape(company)}}}{location_part}\\\\{subtitle}{itemize}
""".rstrip()
        )

    return f"""
\t\\header{{{section_name}}}
\t\\vspace{{1mm}}

{chr(10).join(rendered)}
""".rstrip()


def _build_education(text: str) -> str:
    entries = _parse_education(text)
    if not entries:
        return ""

    rendered = []
    for school, location, details, dates in entries:
        location_part = f"\\hfill {_latex_escape(location)}" if location else ""
        dates_part = f"\\hfill {_latex_escape(dates)}" if dates else ""
        detail_line = f"\n\t{_latex_escape(details)} {dates_part}\\\\" if details or dates else ""
        rendered.append(
            f"""
\t\\textbf{{{_latex_escape(school)}}}{location_part}\\\\{detail_line}
\t\\vspace{{2mm}}
""".rstrip()
        )

    return f"""
\t\\header{{Education}}
{chr(10).join(rendered)}
""".rstrip()


def _parse_skills(text: str) -> list[tuple[str, str]]:
    skills: list[tuple[str, str]] = []
    for line in text.strip().splitlines():
        line = line.strip()
        if ":" not in line:
            continue
        category, value = line.split(":", 1)
        skills.append((category.strip(), value.strip()))
    return skills


def _parse_entries(text: str) -> list[dict]:
    entries: list[dict] = []
    current: dict | None = None
    for raw_line in text.strip().splitlines():
        line = raw_line.strip()
        if not line:
            continue
        if raw_line[:1].isspace() and current and current["bullets"]:
            current["bullets"][-1] = f"{current['bullets'][-1]} {line}"
            continue
        if line.startswith(("- ", "\u2022 ")):
            if current:
                current["bullets"].append(line[2:].strip())
            continue
        if current is None or current["bullets"]:
            if current:
                entries.append(current)
            current = {"title": line, "subtitle": "", "bullets": []}
        elif current["subtitle"]:
            current["bullets"].append(line)
        else:
            current["subtitle"] = line
    if current:
        entries.append(current)
    return entries


def _split_work_entry(entry: dict, profile: dict) -> tuple[str, str, str, str]:
    title = entry.get("title", "").strip()
    subtitle = entry.get("subtitle", "").strip()

    role = ""
    company = title
    location = ""
    dates = ""

    if " at " in title:
        role, company = [part.strip() for part in title.split(" at ", 1)]
    else:
        company, location = _split_company_location(title, profile)

    subtitle_left, subtitle_right = _split_pipe(subtitle)
    if subtitle_right:
        role = role or subtitle_left
        dates = subtitle_right
    else:
        role, dates = _split_role_dates(role or subtitle)

    role = _apply_employment_type(company, role, profile)
    return company or title, location, role, dates


def _split_company_location(text: str, profile: dict) -> tuple[str, str]:
    companies = sorted(
        profile.get("resume_facts", {}).get("preserved_companies", []),
        key=len,
        reverse=True,
    )
    text_lower = text.casefold()
    for company in companies:
        if text_lower.startswith(company.casefold()):
            return company, text[len(company) :].strip(" ,")
    return text, ""


def _apply_employment_type(company: str, role: str, profile: dict) -> str:
    employment_type = _lookup_company_fact(
        company,
        profile.get("resume_facts", {}).get("employment_type_by_company", {}),
    )
    if not employment_type or not role:
        return role

    kind = employment_type.casefold()
    if kind == "contractor":
        if re.search(r"\bcontractor\b", role, re.IGNORECASE):
            return role
        if re.search(r"\bcontract\b", role, re.IGNORECASE):
            return re.sub(r"\bcontract\b", "contractor", role, flags=re.IGNORECASE)
        return f"{role}, contractor"

    if kind in {"full-time", "full time", "fulltime"}:
        if re.search(r"\bfull[- ]?time\b", role, re.IGNORECASE):
            return re.sub(r"\bfull[- ]?time\b", "full-time", role, flags=re.IGNORECASE)
        return f"{role}, full-time"

    if employment_type.casefold() in role.casefold():
        return role
    return f"{role}, {employment_type}"


def _lookup_company_fact(company: str, facts: dict) -> str:
    if not isinstance(facts, dict):
        return ""

    company_key = _normalise_company(company)
    for fact_company, value in facts.items():
        if company_key == _normalise_company(fact_company):
            return str(value)
    return ""


def _normalise_company(company: str) -> str:
    return re.sub(r"[^a-z0-9]+", "", company.casefold())


def _split_pipe(text: str) -> tuple[str, str]:
    if "|" not in text:
        return text, ""
    left, right = text.rsplit("|", 1)
    return left.strip(), right.strip()


def _split_role_dates(text: str) -> tuple[str, str]:
    match = MONTH_PATTERN.search(text)
    if not match:
        return text, ""
    return text[: match.start()].strip(" ,;-"), text[match.start() :].strip()


def _parse_education(text: str) -> list[tuple[str, str, str, str]]:
    lines = [line.strip() for line in text.strip().splitlines() if line.strip()]
    if not lines:
        return []
    if len(lines) == 1:
        left, right = _split_pipe(lines[0])
        return [(left, "", right, "")]

    entries: list[tuple[str, str, str, str]] = []
    for index in range(0, len(lines), 2):
        school_line = lines[index]
        detail_line = lines[index + 1] if index + 1 < len(lines) else ""
        school, location = _split_school_location(school_line)
        details, dates = _split_role_dates(detail_line)
        entries.append((school, location, details, dates))
    return entries


def _split_school_location(text: str) -> tuple[str, str]:
    known_locations = ("Leiden", "Helsinki", "Riga", "Amsterdam", "Rotterdam", "Paris")
    for location in known_locations:
        needle = f" {location}"
        if needle in text:
            school, rest = text.rsplit(needle, 1)
            return school.strip(), f"{location}{rest}".strip(" ,")
    return text, ""


def _ensure_colon(text: str) -> str:
    return text if text.endswith(":") else f"{text}:"


def _latex_escape(text: object) -> str:
    value = str(text)
    value = (
        value.replace("\u00a0", " ")
        .replace("⋅", "·")
        .replace("’", "'")
        .replace("‘", "'")
        .replace("“", '"')
        .replace("”", '"')
        .replace("—", "--")
        .replace("–", "--")
    )
    replacements = {
        "\\": r"\textbackslash{}",
        "&": r"\&",
        "%": r"\%",
        "$": r"\$",
        "#": r"\#",
        "_": r"\_",
        "{": r"\{",
        "}": r"\}",
        "~": r"\textasciitilde{}",
        "^": r"\textasciicircum{}",
        "·": r"$\cdot$",
    }
    return "".join(replacements.get(char, char) for char in value)


def _find_pdflatex() -> str | None:
    if found := shutil.which("pdflatex"):
        return found
    if os.name != "nt":
        return None

    local_app_data = os.environ.get("LOCALAPPDATA")
    candidates = [
        Path("C:/Program Files/MiKTeX/miktex/bin/x64/pdflatex.exe"),
        Path("C:/Program Files/MiKTeX/miktex/bin/pdflatex.exe"),
        Path("C:/Program Files (x86)/MiKTeX/miktex/bin/pdflatex.exe"),
    ]
    if local_app_data:
        candidates.append(Path(local_app_data) / "Programs/MiKTeX/miktex/bin/x64/pdflatex.exe")
    for candidate in candidates:
        if candidate.exists():
            return str(candidate)
    return None
