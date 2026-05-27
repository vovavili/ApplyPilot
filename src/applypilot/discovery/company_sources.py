"""Priority company discovery for curated high-value employers.

This source complements broad job boards by querying explicit company career
surfaces such as Greenhouse, Lever, Workday, and selected custom pages.
"""

from __future__ import annotations

import json
import logging
import re
import sqlite3
from concurrent.futures import ThreadPoolExecutor, as_completed
from dataclasses import dataclass
from datetime import datetime, timezone
from html import unescape
from urllib.parse import quote_plus, urljoin

import httpx
import yaml
from bs4 import BeautifulSoup

from applypilot import config
from applypilot.config import CONFIG_DIR
from applypilot.database import get_connection, init_db
from applypilot.discovery import workday
from applypilot.discovery.workday import REVIEW_STATUS_MANUAL, strip_html
from applypilot.events import emit_error, emit_event

log = logging.getLogger(__name__)

STRATEGY = "priority_company"
REQUIRED_COMPANY_FIELDS = ("name", "platform")
SUPPORTED_PLATFORMS = {"greenhouse", "lever", "workday", "uber", "netflix"}
DEFAULT_HEADERS = {
    "User-Agent": (
        "Mozilla/5.0 (Windows NT 10.0; Win64; x64) AppleWebKit/537.36 "
        "(KHTML, like Gecko) Chrome/126.0.0.0 Safari/537.36"
    ),
    "Accept": "text/html,application/xhtml+xml,application/xml;q=0.9,application/json;q=0.8,*/*;q=0.7",
    "Accept-Language": "en-US,en;q=0.9",
}
NETFLIX_AMSTERDAM_URL = "https://explore.jobs.netflix.net/careers/%2A/amsterdam_netherlands?domain=netflix.com"
UBER_SEARCH_URL = "https://www.uber.com/us/en/careers/list/?query={query_encoded}&location={location_encoded}"
UBER_SEARCH_API_URL = "https://www.uber.com/api/loadSearchJobsResults?localeCode=en"
MISSING_TEXT_VALUES = {"", "none", "null", "nan", "n/a", "na"}


@dataclass(frozen=True)
class SourceResult:
    key: str
    name: str
    found: int = 0
    new: int = 0
    existing: int = 0
    error: str | None = None


# -- Registry ----------------------------------------------------------------


def load_company_sources() -> dict:
    """Load packaged and user priority-company registries."""
    packaged = _load_company_file(CONFIG_DIR / "company_sources.yaml", source="packaged")
    user = _load_company_file(config.APP_DIR / "company_sources.yaml", source="user")
    return {**packaged, **user}


def _load_company_file(path, source: str) -> dict:
    if not path.exists():
        if source == "packaged":
            log.warning("company_sources.yaml not found at %s", path)
        return {}

    data = yaml.safe_load(path.read_text(encoding="utf-8")) or {}
    companies = data.get("companies", {})
    if not isinstance(companies, dict):
        log.warning("Priority company registry %s has no valid 'companies' mapping: %s", source, path)
        return {}
    return _validate_companies(companies, source=source)


def _validate_companies(companies: dict, source: str = "configured") -> dict:
    valid = {}
    for key, company in companies.items():
        if not isinstance(company, dict):
            log.warning("Skipping malformed priority company %s from %s: expected mapping", key, source)
            continue

        missing = [field for field in REQUIRED_COMPANY_FIELDS if not company.get(field)]
        platform = str(company.get("platform", "")).casefold()
        if platform not in SUPPORTED_PLATFORMS:
            missing.append("supported platform")
        elif platform == "greenhouse" and not company.get("board"):
            missing.append("board")
        elif platform == "lever" and not company.get("account"):
            missing.append("account")
        elif platform == "workday" and not company.get("employer_key"):
            missing.append("employer_key")

        if missing:
            log.warning(
                "Skipping malformed priority company %s from %s: missing %s",
                key,
                source,
                ", ".join(dict.fromkeys(missing)),
            )
            continue

        valid[str(key)] = {**company, "platform": platform}
    return valid


def _select_companies(companies: dict, requested: list[str] | str) -> dict:
    if isinstance(requested, str):
        requested = [requested]

    selected = {}
    missing = []
    for item in requested:
        wanted = _normalise_name(item)
        match = None
        for key, company in companies.items():
            if wanted in {_normalise_name(key), _normalise_name(company.get("name", ""))}:
                match = key
                break
        if match:
            selected[match] = companies[match]
        else:
            missing.append(str(item))

    if missing:
        log.warning("Configured priority company sources not found: %s", ", ".join(missing))
    return selected


# -- Public runner -----------------------------------------------------------


def run_priority_company_discovery(companies: dict | None = None, workers: int = 1) -> dict:
    """Run curated company discovery and store normalized jobs."""
    search_cfg = config.load_search_config()
    if not search_cfg.get("priority_company_sources_enabled", True):
        log.info("Priority company discovery disabled by searches.yaml")
        return {"found": 0, "new": 0, "existing": 0, "sources": 0, "errors": 0, "skipped": True}

    requested = search_cfg.get("priority_company_sources") or []
    if not requested:
        log.info("Priority company discovery skipped: no priority_company_sources configured")
        return {"found": 0, "new": 0, "existing": 0, "sources": 0, "errors": 0, "skipped": True}

    if companies is None:
        companies = load_company_sources()
    else:
        companies = _validate_companies(companies, source="provided")
    companies = _select_companies(companies, requested)

    if not companies:
        log.warning("No priority company sources configured or selected.")
        return {"found": 0, "new": 0, "existing": 0, "sources": 0, "errors": 0}

    queries = _load_queries(search_cfg)
    if not queries:
        log.warning("No search queries configured in searches.yaml.")
        return {"found": 0, "new": 0, "existing": 0, "sources": len(companies), "errors": 0}

    accept_locs, reject_locs = workday._load_location_filter(search_cfg)
    location_policy = search_cfg.get("priority_company_location_policy", "recall_first")
    init_db()

    results = _scrape_selected_sources(
        companies,
        queries=queries,
        accept_locs=accept_locs,
        reject_locs=reject_locs,
        location_policy=location_policy,
        search_cfg=search_cfg,
        workers=workers,
    )

    total_found = sum(result.found for result in results)
    total_new = sum(result.new for result in results)
    total_existing = sum(result.existing for result in results)
    errors = sum(1 for result in results if result.error)

    log.info(
        "Priority company discovery done: %d found, %d new, %d existing across %d sources (%d errors)",
        total_found,
        total_new,
        total_existing,
        len(companies),
        errors,
    )
    return {
        "found": total_found,
        "new": total_new,
        "existing": total_existing,
        "sources": len(companies),
        "errors": errors,
    }


def _scrape_selected_sources(
    companies: dict,
    *,
    queries: list[str],
    accept_locs: list[str],
    reject_locs: list[str],
    location_policy: str,
    search_cfg: dict,
    workers: int = 1,
) -> list[SourceResult]:
    args = {
        "queries": queries,
        "accept_locs": accept_locs,
        "reject_locs": reject_locs,
        "location_policy": location_policy,
        "search_cfg": search_cfg,
    }

    if workers > 1 and len(companies) > 1:
        results: list[SourceResult] = []
        with ThreadPoolExecutor(max_workers=min(workers, len(companies))) as pool:
            futures = {
                pool.submit(_scrape_and_store_source, key, company, **args): key for key, company in companies.items()
            }
            for future in as_completed(futures):
                results.append(future.result())
        return results

    return [_scrape_and_store_source(key, company, **args) for key, company in companies.items()]


def _scrape_and_store_source(
    key: str,
    company: dict,
    *,
    queries: list[str],
    accept_locs: list[str],
    reject_locs: list[str],
    location_policy: str,
    search_cfg: dict,
) -> SourceResult:
    name = str(company.get("name") or key)
    try:
        jobs = _scrape_source(
            key,
            company,
            queries=queries,
            accept_locs=accept_locs,
            reject_locs=reject_locs,
            location_policy=location_policy,
            search_cfg=search_cfg,
        )
        new, existing = store_priority_company_jobs(get_connection(), jobs)
        log.info("%s priority source: %d found, %d new, %d existing", name, len(jobs), new, existing)
        emit_event(
            "priority_company_source_finished",
            stage="discover",
            source="priority_company",
            status="ok",
            company_key=key,
            company=name,
            platform=company.get("platform"),
            found=len(jobs),
            new=new,
            existing=existing,
        )
        return SourceResult(key=key, name=name, found=len(jobs), new=new, existing=existing)
    except Exception as exc:
        log.error("%s priority source failed: %s", name, exc)
        emit_error(
            "priority_company_source_failed",
            exc,
            stage="discover",
            source="priority_company",
            company_key=key,
            company=name,
            platform=company.get("platform"),
        )
        return SourceResult(key=key, name=name, error=str(exc))


def _scrape_source(
    key: str,
    company: dict,
    *,
    queries: list[str],
    accept_locs: list[str],
    reject_locs: list[str],
    location_policy: str,
    search_cfg: dict,
) -> list[dict]:
    platform = company["platform"]
    if platform == "greenhouse":
        return _scrape_greenhouse(key, company, queries, accept_locs, reject_locs, location_policy, search_cfg)
    if platform == "lever":
        return _scrape_lever(key, company, queries, accept_locs, reject_locs, location_policy, search_cfg)
    if platform == "workday":
        return _scrape_workday_bridge(key, company, queries, accept_locs, reject_locs, location_policy, search_cfg)
    if platform == "netflix":
        return _scrape_netflix(key, company, queries, accept_locs, reject_locs, location_policy, search_cfg)
    if platform == "uber":
        return _scrape_uber(key, company, queries, accept_locs, reject_locs, location_policy, search_cfg)
    raise ValueError(f"Unsupported priority company platform: {platform}")


# -- Platform adapters -------------------------------------------------------


def _scrape_greenhouse(
    key: str,
    company: dict,
    queries: list[str],
    accept_locs: list[str],
    reject_locs: list[str],
    location_policy: str,
    search_cfg: dict | None = None,
) -> list[dict]:
    board = company["board"]
    data = _http_get_json(f"https://boards-api.greenhouse.io/v1/boards/{board}/jobs?content=true")
    jobs = []
    for item in data.get("jobs", []):
        full_description = strip_html(item.get("content", ""))
        title = item.get("title", "")
        if not _matches_queries([title, full_description], queries):
            continue

        location = _greenhouse_location(item)
        job = _make_job(
            source_key=key,
            company=company,
            title=title,
            url=item.get("absolute_url"),
            location=location,
            full_description=full_description,
        )
        jobs.extend(_triaged_job(job, accept_locs, reject_locs, location_policy, search_cfg))
    return jobs


def _scrape_lever(
    key: str,
    company: dict,
    queries: list[str],
    accept_locs: list[str],
    reject_locs: list[str],
    location_policy: str,
    search_cfg: dict | None = None,
) -> list[dict]:
    account = company["account"]
    data = _http_get_json(f"https://api.lever.co/v0/postings/{account}?mode=json")
    jobs = []
    for item in data if isinstance(data, list) else []:
        full_description = _lever_description(item)
        title = item.get("text", "")
        if not _matches_queries([title, full_description], queries):
            continue

        location = str((item.get("categories") or {}).get("location") or "").strip()
        job = _make_job(
            source_key=key,
            company=company,
            title=title,
            url=item.get("hostedUrl") or item.get("applyUrl"),
            application_url=item.get("applyUrl") or item.get("hostedUrl"),
            location=location,
            full_description=full_description,
            salary=_lever_salary(item),
        )
        jobs.extend(_triaged_job(job, accept_locs, reject_locs, location_policy, search_cfg))
    return jobs


def _scrape_workday_bridge(
    key: str,
    company: dict,
    queries: list[str],
    accept_locs: list[str],
    reject_locs: list[str],
    location_policy: str,
    search_cfg: dict | None = None,
) -> list[dict]:
    employer_key = company["employer_key"]
    employers = workday.load_employers()
    employer = employers.get(employer_key)
    if not employer:
        raise ValueError(f"Workday employer key not found: {employer_key}")

    jobs: list[dict] = []
    for query in queries:
        found = workday.search_employer(
            employer_key,
            employer,
            query,
            location_filter=True,
            accept_locs=accept_locs,
            reject_locs=reject_locs,
            location_policy=location_policy,
            search_cfg=search_cfg,
        )
        detailed = workday.fetch_details(
            employer,
            found,
            location_filter=True,
            accept_locs=accept_locs,
            reject_locs=reject_locs,
            location_policy=location_policy,
            search_cfg=search_cfg,
        )
        for item in detailed:
            url = (
                item.get("apply_url") or f"{employer['base_url']}/{employer['site_id']}{item.get('external_path', '')}"
            )
            job = _make_job(
                source_key=key,
                company=company,
                title=item.get("title"),
                url=url,
                location=item.get("location"),
                full_description=item.get("full_description", ""),
                application_url=url,
            )
            job["location_decision"] = item.get("location_decision")
            job["location_reason"] = item.get("location_reason")
            jobs.append(job)
    return _dedupe_jobs(jobs)


def _scrape_netflix(
    key: str,
    company: dict,
    queries: list[str],
    accept_locs: list[str],
    reject_locs: list[str],
    location_policy: str,
    search_cfg: dict | None = None,
) -> list[dict]:
    url = company.get("url") or NETFLIX_AMSTERDAM_URL
    html = _http_get_text(url)
    jobs = _scrape_netflix_static(key, company, html, queries, accept_locs, reject_locs, location_policy, search_cfg)
    if not jobs:
        jobs = _scrape_rendered_job_links(
            url,
            source_key=key,
            company=company,
            queries=queries,
            link_substring="/careers/job/",
            accept_locs=accept_locs,
            reject_locs=reject_locs,
            location_policy=location_policy,
            search_cfg=search_cfg,
        )
    return _dedupe_jobs(jobs)


def _scrape_netflix_static(
    key: str,
    company: dict,
    html: str,
    queries: list[str],
    accept_locs: list[str],
    reject_locs: list[str],
    location_policy: str,
    search_cfg: dict | None = None,
) -> list[dict]:
    text = unescape(html)
    positions = _extract_json_array_after_key(text, '"positions"')
    jobs = []
    for item in positions:
        title = item.get("posting_name") or item.get("name") or ""
        location = _clean_location(item.get("location") or ", ".join(item.get("locations") or []))
        if not _matches_queries([title, item.get("department", ""), item.get("business_unit", "")], queries):
            continue

        url = item.get("canonicalPositionUrl")
        full_description = strip_html(item.get("job_description", ""))
        if url and not full_description:
            full_description = _fetch_netflix_description(url)
        job = _make_job(
            source_key=key,
            company=company,
            title=title,
            url=url,
            location=location,
            full_description=full_description,
        )
        jobs.extend(_triaged_job(job, accept_locs, reject_locs, location_policy, search_cfg))
    return jobs


def _scrape_uber(
    key: str,
    company: dict,
    queries: list[str],
    accept_locs: list[str],
    reject_locs: list[str],
    location_policy: str,
    search_cfg: dict,
) -> list[dict]:
    jobs: list[dict] = []
    location = _default_location(search_cfg)
    template = company.get("search_url_template") or UBER_SEARCH_URL

    for query in queries:
        api_jobs = _scrape_uber_api(
            key,
            company,
            query,
            location,
            accept_locs,
            reject_locs,
            location_policy,
            search_cfg,
        )
        if api_jobs:
            jobs.extend(api_jobs)
            continue

        url = template.format(query_encoded=quote_plus(query), location_encoded=quote_plus(location))
        html = _http_get_text(url)
        parsed = _scrape_uber_static(
            key,
            company,
            html,
            queries=[query],
            accept_locs=accept_locs,
            reject_locs=reject_locs,
            location_policy=location_policy,
            search_cfg=search_cfg,
        )
        if not parsed:
            parsed = _scrape_rendered_job_links(
                url,
                source_key=key,
                company=company,
                queries=[query],
                link_substring="/careers/list/",
                accept_locs=accept_locs,
                reject_locs=reject_locs,
                location_policy=location_policy,
                search_cfg=search_cfg,
            )
        jobs.extend(parsed)

    return _dedupe_jobs(jobs)


def _scrape_uber_api(
    key: str,
    company: dict,
    query: str,
    location: str,
    accept_locs: list[str],
    reject_locs: list[str],
    location_policy: str,
    search_cfg: dict | None = None,
) -> list[dict]:
    jobs = []
    location_filters = _uber_location_filters(location)
    require_title_match = not location_filters
    for page in range(3):
        params: dict[str, object] = {"query": query}
        if location_filters:
            params["location"] = location_filters
        data = _http_post_json(
            UBER_SEARCH_API_URL,
            {
                "limit": 20,
                "page": page,
                "params": params,
            },
            referer=UBER_SEARCH_URL.format(query_encoded=quote_plus(query), location_encoded=quote_plus(location)),
        )
        results = ((data or {}).get("data") or {}).get("results") or []
        if not results:
            break

        for item in results:
            title = item.get("title", "")
            if require_title_match and not _matches_queries([title], [query]):
                continue
            job = _make_job(
                source_key=key,
                company=company,
                title=title,
                url=f"https://www.uber.com/us/en/careers/list/{item.get('id')}/" if item.get("id") else "",
                location=_uber_location(item),
                full_description=item.get("description", ""),
            )
            jobs.extend(_triaged_job(job, accept_locs, reject_locs, location_policy, search_cfg))

        total = ((data or {}).get("data") or {}).get("totalResults") or {}
        total_low = int(total.get("low") or 0) if isinstance(total, dict) else 0
        if (page + 1) * 20 >= total_low:
            break
    return jobs


def _scrape_uber_static(
    key: str,
    company: dict,
    html: str,
    queries: list[str],
    accept_locs: list[str],
    reject_locs: list[str],
    location_policy: str,
    search_cfg: dict | None = None,
) -> list[dict]:
    soup = BeautifulSoup(html, "html.parser")
    jobs = []
    for anchor in soup.select('a[href*="/careers/list/"]'):
        href = anchor.get("href")
        title = anchor.get_text(" ", strip=True)
        if not href or not _looks_like_uber_job_url(href) or not _matches_queries([title], queries):
            continue

        row_text = _nearest_text(anchor)
        location = _extract_locationish_text(row_text)
        job = _make_job(
            source_key=key,
            company=company,
            title=title,
            url=urljoin("https://www.uber.com", href),
            location=location,
            full_description="",
        )
        jobs.extend(_triaged_job(job, accept_locs, reject_locs, location_policy, search_cfg))
    return jobs


# -- Storage -----------------------------------------------------------------


def store_priority_company_jobs(conn: sqlite3.Connection, jobs: list[dict]) -> tuple[int, int]:
    now = datetime.now(timezone.utc).isoformat()
    new = 0
    existing = 0

    for job in jobs:
        url = job.get("url")
        if not url:
            continue

        full_description = str(job.get("full_description") or "").strip()
        short_desc = full_description[:500] if full_description else job.get("description")
        detail_scraped_at = now if full_description else None
        review_status = REVIEW_STATUS_MANUAL if job.get("location_decision") == "manual_review" else None
        review_reason = (
            f"priority_company_location:{job.get('location_reason', 'manual_review')}" if review_status else None
        )

        try:
            conn.execute(
                "INSERT INTO jobs (url, title, salary, description, location, site, strategy, "
                "discovered_at, full_description, application_url, detail_scraped_at, detail_error, "
                "review_status, review_reason) "
                "VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?)",
                (
                    url,
                    job.get("title"),
                    job.get("salary"),
                    short_desc,
                    job.get("location"),
                    job.get("company"),
                    STRATEGY,
                    now,
                    full_description or None,
                    job.get("application_url") or url,
                    detail_scraped_at,
                    job.get("detail_error"),
                    review_status,
                    review_reason,
                ),
            )
            new += 1
        except sqlite3.IntegrityError:
            existing += 1

    conn.commit()
    return new, existing


# -- Helpers -----------------------------------------------------------------


def _http_get_json(url: str):
    with httpx.Client(headers=DEFAULT_HEADERS, follow_redirects=True, timeout=30) as client:
        response = client.get(url)
        response.raise_for_status()
        return response.json()


def _http_get_text(url: str) -> str:
    with httpx.Client(headers=DEFAULT_HEADERS, follow_redirects=True, timeout=30) as client:
        response = client.get(url)
        response.raise_for_status()
        return response.text


def _http_post_json(url: str, payload: dict, *, referer: str | None = None):
    headers = {
        **DEFAULT_HEADERS,
        "Accept": "application/json, text/plain, */*",
        "Content-Type": "application/json",
        "Origin": "https://www.uber.com",
        "x-csrf-token": "x",
    }
    if referer:
        headers["Referer"] = referer
    with httpx.Client(headers=headers, follow_redirects=True, timeout=30) as client:
        response = client.post(url, json=payload)
        response.raise_for_status()
        return response.json()


def _load_queries(search_cfg: dict) -> list[str]:
    max_tier = int(search_cfg.get("priority_company_max_tier", 3))
    queries_cfg = search_cfg.get("queries", [])
    queries = [q["query"] for q in queries_cfg if q.get("tier", 99) <= max_tier and q.get("query")]
    if not queries:
        queries = [q["query"] for q in queries_cfg if q.get("query")]
    return list(dict.fromkeys(queries))


def _make_job(
    *,
    source_key: str,
    company: dict,
    title: object,
    url: object,
    location: object,
    full_description: object = "",
    application_url: object | None = None,
    salary: object | None = None,
) -> dict:
    url_text = _optional_text(url)
    application_url_text = _optional_text(application_url) or url_text or ""
    return {
        "source_key": source_key,
        "company": company.get("name"),
        "title": str(title or "").strip(),
        "url": url_text or "",
        "application_url": application_url_text,
        "salary": str(salary).strip() if salary else None,
        "location": _clean_location(location),
        "full_description": str(full_description or "").strip(),
    }


def _optional_text(value: object) -> str | None:
    text = str(value or "").strip()
    if text.casefold() in MISSING_TEXT_VALUES:
        return None
    return text


def _triaged_job(
    job: dict,
    accept_locs: list[str],
    reject_locs: list[str],
    location_policy: str,
    search_cfg: dict | None = None,
) -> list[dict]:
    triage = workday._triage_location(
        job.get("location"),
        accept_locs,
        reject_locs,
        policy=location_policy,
        search_cfg=search_cfg,
    )
    if triage.decision == "reject":
        return []
    return [{**job, "location_decision": triage.decision, "location_reason": triage.reason}]


def _greenhouse_location(item: dict) -> str:
    location = item.get("location") or {}
    if isinstance(location, dict) and location.get("name"):
        return _clean_location(location["name"])

    offices = item.get("offices") or []
    office_names = [office.get("name") for office in offices if isinstance(office, dict) and office.get("name")]
    return _clean_location(" | ".join(office_names))


def _lever_description(item: dict) -> str:
    parts = [item.get("descriptionPlain"), item.get("additionalPlain")]
    for section in item.get("lists") or []:
        if not isinstance(section, dict):
            continue
        parts.append(section.get("text"))
        parts.append(strip_html(section.get("content", "")))
    return "\n\n".join(str(part).strip() for part in parts if str(part or "").strip())


def _lever_salary(item: dict) -> str | None:
    salary = item.get("salaryRange")
    if isinstance(salary, dict):
        parts = [salary.get("currency"), salary.get("min"), salary.get("max"), salary.get("interval")]
        return " ".join(str(part) for part in parts if part)
    return None


def _uber_location(item: dict) -> str:
    locations = item.get("allLocations") or []
    if not locations and isinstance(item.get("location"), dict):
        locations = [item["location"]]
    return _clean_location(" | ".join(_format_uber_location(location) for location in locations))


def _format_uber_location(location: dict) -> str:
    if not isinstance(location, dict):
        return ""
    parts = [
        location.get("city"),
        location.get("region"),
        location.get("countryName") or location.get("country"),
    ]
    return ", ".join(str(part).strip() for part in parts if str(part or "").strip())


def _uber_location_filters(location: str) -> list[dict]:
    text = location.casefold()
    if "amsterdam" in text or "netherlands" in text or "nederland" in text:
        return [{"city": "Amsterdam", "country": "NLD", "countryName": "Netherlands"}]
    return []


def _fetch_netflix_description(url: str) -> str:
    try:
        html = _http_get_text(url)
    except Exception as exc:
        log.info("Netflix detail fetch failed for %s: %s", url, exc)
        return ""

    soup = BeautifulSoup(html, "html.parser")
    for script in soup.find_all("script", {"type": "application/ld+json"}):
        raw = script.string or script.get_text()
        try:
            data = json.loads(raw)
        except json.JSONDecodeError:
            continue
        if isinstance(data, dict) and data.get("@type") == "JobPosting":
            return strip_html(data.get("description", ""))
    return ""


def _extract_json_array_after_key(text: str, key: str) -> list[dict]:
    pos = text.find(key)
    if pos < 0:
        return []
    start = text.find("[", pos)
    if start < 0:
        return []

    decoder = json.JSONDecoder()
    try:
        value, _ = decoder.raw_decode(text[start:])
    except json.JSONDecodeError:
        return []
    return value if isinstance(value, list) else []


def _scrape_rendered_job_links(
    url: str,
    *,
    source_key: str,
    company: dict,
    queries: list[str],
    link_substring: str,
    accept_locs: list[str],
    reject_locs: list[str],
    location_policy: str,
    search_cfg: dict | None = None,
) -> list[dict]:
    try:
        from playwright.sync_api import sync_playwright

        from applypilot.playwright_utils import launch_chromium
    except Exception as exc:
        log.info("%s rendered fallback unavailable: %s", company.get("name"), exc)
        return []

    try:
        with sync_playwright() as p:
            browser = launch_chromium(p.chromium, headless=True)
            page = browser.new_page()
            page.goto(url, wait_until="networkidle", timeout=45_000)
            page.wait_for_timeout(2_000)
            raw_jobs = page.eval_on_selector_all(
                f'a[href*="{link_substring}"]',
                """anchors => anchors.map(anchor => {
                    const row = anchor.closest('li, article, section, div');
                    return {
                        title: (anchor.innerText || anchor.textContent || '').trim(),
                        url: anchor.href,
                        text: row ? (row.innerText || row.textContent || '').trim() : ''
                    };
                })""",
            )
            browser.close()
    except Exception as exc:
        log.info("%s rendered fallback failed: %s", company.get("name"), exc)
        return []

    jobs = []
    for item in raw_jobs:
        title = str(item.get("title") or "").strip()
        href = str(item.get("url") or "").strip()
        if not title or not href or not _matches_queries([title], queries):
            continue
        if company.get("platform") == "uber" and not _looks_like_uber_job_url(href):
            continue
        location = _extract_locationish_text(str(item.get("text") or ""))
        job = _make_job(
            source_key=source_key,
            company=company,
            title=title,
            url=href,
            location=location,
        )
        jobs.extend(_triaged_job(job, accept_locs, reject_locs, location_policy, search_cfg))
    return _dedupe_jobs(jobs)


def _matches_queries(values: list[object], queries: list[str]) -> bool:
    if not queries:
        return True

    title = str(values[0] if values else "").casefold()
    title_tokens = set(re.findall(r"[a-z0-9]+", title))
    for query in queries:
        query_text = str(query or "").strip().casefold()
        if not query_text:
            continue
        query_tokens = re.findall(r"[a-z0-9]+", query_text)
        if query_text in title or (len(query_tokens) >= 3 and all(token in title_tokens for token in query_tokens)):
            return True
    return False


def _clean_location(value: object) -> str:
    text = str(value or "").strip()
    text = re.sub(r"\s*,\s*", ", ", text)
    text = re.sub(r"\s+", " ", text)
    return text


def _extract_locationish_text(text: str) -> str:
    lines = [_clean_location(line) for line in re.split(r"[\n\r|]+", text) if _clean_location(line)]
    for line in lines:
        lower = line.casefold()
        triage = workday._triage_location(line, [], [], policy="recall_first")
        if triage.decision == "accept" or any(term in lower for term in ("amsterdam", "netherlands", "remote")):
            return line
    return ""


def _nearest_text(anchor) -> str:
    node = anchor
    for _ in range(4):
        node = node.parent
        if node is None:
            break
        text = node.get_text("\n", strip=True)
        if text and len(text) > len(anchor.get_text(" ", strip=True)):
            return text
    return anchor.get_text("\n", strip=True)


def _looks_like_uber_job_url(url: str) -> bool:
    path = url.casefold()
    return "/careers/list/" in path and re.search(r"/careers/list/\d+", path) is not None


def _default_location(search_cfg: dict) -> str:
    locations = search_cfg.get("locations") or []
    if locations and isinstance(locations[0], dict) and locations[0].get("location"):
        return str(locations[0]["location"])
    defaults = search_cfg.get("defaults") or {}
    return str(defaults.get("location") or "Amsterdam")


def _dedupe_jobs(jobs: list[dict]) -> list[dict]:
    seen = set()
    deduped = []
    for job in jobs:
        url = job.get("url")
        if not url or url in seen:
            continue
        seen.add(url)
        deduped.append(job)
    return deduped


def _normalise_name(value: object) -> str:
    return re.sub(r"[^a-z0-9]+", "", str(value).casefold())
