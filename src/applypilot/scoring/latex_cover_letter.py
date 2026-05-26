"""LaTeX cover-letter renderer using the user's moderncv letter style."""

from __future__ import annotations

import re

from applypilot.scoring.latex_resume import _latex_escape


BROAD_JOB_BOARD_SITES = {
    "indeed",
    "linkedin",
    "glassdoor",
    "zip_recruiter",
    "ziprecruiter",
}


def build_latex_cover_letter(text: str, profile: dict | None = None, job: dict | None = None) -> str:
    """Build a one-page moderncv-style cover letter from plain text."""
    profile = profile or {}
    job = job or {}
    personal = profile.get("personal", {})
    name_parts = _name_parts(str(personal.get("full_name") or ""))
    full_name = " ".join(part for part in name_parts if part).strip() or str(personal.get("preferred_name") or "")
    title = (
        profile.get("experience", {}).get("target_role")
        or profile.get("experience", {}).get("current_title")
        or "Data Engineer"
    )
    address_line_1 = _country_label(str(personal.get("country") or "The Netherlands"))
    address_line_2 = str(personal.get("city") or "").strip() or "The Hague"
    address_line_3 = (
        str(profile.get("work_authorization", {}).get("work_permit_type") or "").strip() or "EU/Latvian citizen"
    )
    company = _display_company(job)
    paragraphs = _body_paragraphs(text, full_name)

    body = "\n\n".join(f"\\noindent {_latex_escape(paragraph)}" for paragraph in paragraphs)

    return rf"""
\documentclass[11pt,a4paper,roman]{{moderncv}}
\usepackage[english]{{babel}}
\moderncvstyle{{classic}}
\moderncvcolor{{green}}
\usepackage[utf8]{{inputenc}}
\usepackage[T1]{{fontenc}}
\usepackage[scale=0.75]{{geometry}}
\usepackage{{lmodern}}
\nopagenumbers{{}}

\name{{{_latex_escape(name_parts[0])}}}{{{_latex_escape(name_parts[1])}}}
\title{{{_latex_escape(title)}}}
\address{{{_latex_escape(address_line_1)}}}{{{_latex_escape(address_line_2)}}}{{{_latex_escape(address_line_3)}}}
\phone[mobile]{{{_latex_escape(personal.get("phone", ""))}}}
\email{{{_latex_escape(personal.get("email", ""))}}}

\begin{{document}}
\recipient{{Hiring Manager}}{{{_latex_escape(company)}}}
\date{{\today}}
\opening{{Dear Hiring Manager,}}
\closing{{Kind regards,}}

\makelettertitle

{body}

\vspace{{0.2cm}}

\noindent Kind regards,

\vspace{{-0.2cm}}

\noindent\textbf{{{_latex_escape(full_name)}}}

\end{{document}}
""".strip()


def _name_parts(full_name: str) -> tuple[str, str]:
    parts = full_name.strip().split()
    if not parts:
        return "Vladimir", "Vilimaitis"
    if len(parts) == 1:
        return parts[0], ""
    return " ".join(parts[:-1]), parts[-1]


def _country_label(country: str) -> str:
    if country.strip().casefold() == "netherlands":
        return "The Netherlands"
    return country.strip() or "The Netherlands"


def _display_company(job: dict) -> str:
    company = str(job.get("company") or "").strip()
    if company:
        return company

    title = str(job.get("title") or "").strip()
    site = str(job.get("site") or "").strip()
    if site.casefold() in BROAD_JOB_BOARD_SITES:
        if "|" in title:
            inferred = title.rsplit("|", 1)[1].strip()
            if inferred:
                return inferred
        return "Hiring Team"
    return site or "Hiring Team"


def _body_paragraphs(text: str, signoff_name: str) -> list[str]:
    lines = [line.strip() for line in text.strip().splitlines()]
    while lines and not lines[0]:
        lines.pop(0)
    if lines and lines[0].casefold().startswith("dear "):
        lines.pop(0)

    while lines and not lines[-1]:
        lines.pop()
    while lines and _is_closing_line(lines[-1], signoff_name):
        lines.pop()
        while lines and not lines[-1]:
            lines.pop()

    paragraphs: list[str] = []
    current: list[str] = []
    for line in lines:
        if not line:
            if current:
                paragraphs.append(" ".join(current).strip())
                current = []
            continue
        current.append(line)
    if current:
        paragraphs.append(" ".join(current).strip())

    return paragraphs or [text.strip()]


def _is_closing_line(line: str, signoff_name: str) -> bool:
    normalized = re.sub(r"[^\w\s]", "", line).strip().casefold()
    signoff = re.sub(r"[^\w\s]", "", signoff_name).strip().casefold()
    return normalized in {signoff, "kind regards", "regards", "sincerely"}
