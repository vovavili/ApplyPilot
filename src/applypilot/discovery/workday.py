"""Workday ATS direct API scraper: searches employer career portals.

Scrapes Workday-powered career sites (TD, RBC, NVIDIA, Salesforce, etc.)
via the undocumented CXS JSON API. Zero LLM, zero browser -- pure HTTP.

Employer registry is loaded from config/employers.yaml instead of being
hardcoded. Supports sequential search + detail fetching with proxy.
"""

import json
import logging
import re
import sqlite3
import time
import urllib.request
from concurrent.futures import ThreadPoolExecutor, as_completed
from datetime import datetime, timezone
from html.parser import HTMLParser

import yaml

from applypilot import config
from applypilot.config import CONFIG_DIR
from applypilot.database import get_connection, init_db
from applypilot.discovery.location import LocationTriage, triage_location
from applypilot.events import emit_error, emit_event

log = logging.getLogger(__name__)

REVIEW_STATUS_MANUAL = "manual_review"

REMOTE_TERMS = (
    "remote",
    "virtual",
    "work from home",
    "wfh",
    "distributed",
    "anywhere",
    "home based",
    "home-based",
)
EUROPE_TERMS = (
    "europe",
    "european union",
    "emea",
)
SHORT_EU_TERMS = ("eu",)
NL_TERMS = (
    "amsterdam",
    "rotterdam",
    "utrecht",
    "the hague",
    "den haag",
    "delft",
    "leiden",
    "south holland",
    "zuid-holland",
)
VAGUE_LOCATION_TERMS = (
    "multiple locations",
    "various locations",
    "multiple",
    "global",
    "worldwide",
    "flexible",
    "hybrid",
)
WORKDAY_REQUIRED_EMPLOYER_FIELDS = ("name", "tenant", "site_id", "base_url")


# -- Employer registry from YAML --------------------------------------------


def load_employers() -> dict:
    """Load packaged and user Workday employer registries."""
    packaged = _load_employer_file(CONFIG_DIR / "employers.yaml", source="packaged")
    user = _load_employer_file(config.APP_DIR / "workday_employers.yaml", source="user")
    return {**packaged, **user}


def _load_employer_file(path, source: str) -> dict:
    """Load and validate a Workday employer YAML file."""
    if not path.exists():
        if source == "packaged":
            log.warning("employers.yaml not found at %s", path)
        return {}

    data = yaml.safe_load(path.read_text(encoding="utf-8")) or {}
    employers = data.get("employers", {})
    if not isinstance(employers, dict):
        log.warning("Workday employer registry %s has no valid 'employers' mapping: %s", source, path)
        return {}
    return _validate_employers(employers, source=source)


def _validate_employers(employers: dict, source: str = "configured") -> dict:
    """Drop malformed employer entries and keep valid Workday definitions."""
    valid = {}
    for key, employer in employers.items():
        if not isinstance(employer, dict):
            log.warning("Skipping malformed Workday employer %s from %s: expected mapping", key, source)
            continue

        missing = [field for field in WORKDAY_REQUIRED_EMPLOYER_FIELDS if not employer.get(field)]
        if missing:
            log.warning(
                "Skipping malformed Workday employer %s from %s: missing %s",
                key,
                source,
                ", ".join(missing),
            )
            continue
        valid[str(key)] = employer
    return valid


# -- Location filtering from search config -----------------------------------


def _load_location_filter(search_cfg: dict | None = None):
    """Load location accept/reject lists from search config."""
    if search_cfg is None:
        search_cfg = config.load_search_config()

    accept = search_cfg.get("location_accept", [])
    reject = search_cfg.get("location_reject_non_remote", [])
    return accept, reject


def _location_ok(
    location: str | None,
    accept: list[str],
    reject: list[str],
    search_cfg: dict | None = None,
) -> bool:
    """Backward-compatible boolean wrapper around Workday location triage."""
    return _triage_location(location, accept, reject, search_cfg=search_cfg).decision == "accept"


def _triage_location(
    location: str | None,
    accept: list[str],
    reject: list[str],
    *,
    policy: str = "recall_first",
    search_cfg: dict | None = None,
) -> LocationTriage:
    """Classify a Workday location as accepted, manual-review, or rejected."""
    return triage_location(location, accept, reject, policy=policy, search_cfg=search_cfg)


def _ambiguous_location(policy: str, reason: str) -> LocationTriage:
    if policy == "strict":
        return LocationTriage("reject", reason)
    if policy == "balanced" and reason != "ambiguous_location":
        return LocationTriage("reject", reason)
    return LocationTriage("manual_review", reason)


def _contains_any(text: str, terms: list[str] | tuple[str, ...]) -> bool:
    return any(str(term).strip().casefold() in text for term in terms if str(term).strip())


def _contains_accept_term(text: str, terms: list[str]) -> bool:
    for term in terms:
        value = str(term).strip().casefold()
        if not value:
            continue
        if len(value) <= 3:
            if re.search(rf"\b{re.escape(value)}\b", text):
                return True
        elif value in text:
            return True
    return False


def _contains_short_term(text: str, terms: tuple[str, ...]) -> bool:
    return any(re.search(rf"\b{re.escape(term)}\b", text) for term in terms)


# -- HTML stripper -----------------------------------------------------------


class _HTMLStripper(HTMLParser):
    """Strip HTML tags, keep text content."""

    def __init__(self):
        super().__init__()
        self._parts: list[str] = []
        self._skip = False

    def handle_starttag(self, tag, attrs):
        if tag in ("script", "style"):
            self._skip = True
        elif tag in ("br", "p", "div", "li", "tr", "h1", "h2", "h3", "h4", "h5", "h6"):
            self._parts.append("\n")

    def handle_endtag(self, tag):
        if tag in ("script", "style"):
            self._skip = False
        elif tag in ("p", "div", "li", "tr"):
            self._parts.append("\n")

    def handle_data(self, data):
        if not self._skip:
            self._parts.append(data)

    def get_text(self) -> str:
        text = "".join(self._parts)
        text = re.sub(r"[^\S\n]+", " ", text)
        text = re.sub(r"\n{3,}", "\n\n", text)
        return text.strip()


def strip_html(html: str) -> str:
    """Convert HTML to plain text."""
    if not html:
        return ""
    stripper = _HTMLStripper()
    stripper.feed(html)
    return stripper.get_text()


# -- Proxy -------------------------------------------------------------------

_opener = None


def setup_proxy(proxy_str: str | None) -> None:
    """Configure a global urllib opener with proxy support."""
    global _opener
    if not proxy_str:
        _opener = urllib.request.build_opener()
        return

    parts = proxy_str.split(":")
    if len(parts) == 4:
        host, port, user, passwd = parts
        proxy_url = f"http://{user}:{passwd}@{host}:{port}"
    elif len(parts) == 2:
        proxy_url = f"http://{parts[0]}:{parts[1]}"
    else:
        log.warning("Proxy format not recognized: %s (expected host:port:user:pass or host:port)", proxy_str)
        _opener = urllib.request.build_opener()
        return

    proxy_handler = urllib.request.ProxyHandler(
        {
            "http": proxy_url,
            "https": proxy_url,
        }
    )
    _opener = urllib.request.build_opener(proxy_handler)
    log.info("Proxy configured: %s:%s", parts[0], parts[1])


def _urlopen(req, timeout=30):
    """Open a URL using the configured opener (with or without proxy)."""
    if _opener:
        return _opener.open(req, timeout=timeout)
    return urllib.request.urlopen(req, timeout=timeout)


# -- Workday API -------------------------------------------------------------


def workday_search(employer: dict, search_text: str, limit: int = 20, offset: int = 0) -> dict:
    """Search jobs via Workday CXS API. Returns JSON with total + jobPostings."""
    url = f"{employer['base_url']}/wday/cxs/{employer['tenant']}/{employer['site_id']}/jobs"
    payload = json.dumps(
        {
            "appliedFacets": {},
            "limit": limit,
            "offset": offset,
            "searchText": search_text,
        }
    ).encode()

    req = urllib.request.Request(url, data=payload, method="POST")
    req.add_header("Content-Type", "application/json")
    req.add_header("Accept", "application/json")
    req.add_header("User-Agent", "Mozilla/5.0 (Windows NT 10.0; Win64; x64) AppleWebKit/537.36")

    with _urlopen(req, timeout=30) as resp:
        return json.loads(resp.read())


def workday_detail(employer: dict, external_path: str) -> dict:
    """Fetch full job detail via Workday CXS API."""
    url = f"{employer['base_url']}/wday/cxs/{employer['tenant']}/{employer['site_id']}{external_path}"

    req = urllib.request.Request(url)
    req.add_header("Accept", "application/json")
    req.add_header("User-Agent", "Mozilla/5.0 (Windows NT 10.0; Win64; x64) AppleWebKit/537.36")

    with _urlopen(req, timeout=30) as resp:
        return json.loads(resp.read())


# -- Search + paginate -------------------------------------------------------


def search_employer(
    employer_key: str,
    employer: dict,
    search_text: str,
    location_filter: bool = True,
    max_results: int = 0,
    accept_locs: list[str] | None = None,
    reject_locs: list[str] | None = None,
    location_policy: str = "recall_first",
    search_cfg: dict | None = None,
) -> list[dict]:
    """Search an employer and keep accepted or review-worthy location matches."""
    log.info('%s: searching "%s"...', employer["name"], search_text)

    all_jobs: list[dict] = []
    offset = 0
    page_size = 20
    max_pages = 25  # Cap at 500 results
    total = None

    while True:
        try:
            data = workday_search(employer, search_text, limit=page_size, offset=offset)
        except Exception as e:
            log.error("%s: API error at offset %d: %s", employer["name"], offset, e)
            break

        if total is None:
            total = data.get("total", 0)
            log.info("%s: %d total results", employer["name"], total)

        postings = data.get("jobPostings", [])
        if not postings:
            break

        for j in postings:
            loc = j.get("locationsText", "")
            triage = LocationTriage("accept", "location_filter_disabled")
            if location_filter and accept_locs is not None and reject_locs is not None:
                triage = _triage_location(
                    loc,
                    accept_locs,
                    reject_locs,
                    policy=location_policy,
                    search_cfg=search_cfg,
                )
                if triage.decision == "reject":
                    continue

            all_jobs.append(
                {
                    "title": j.get("title", ""),
                    "location": loc,
                    "posted": j.get("postedOn", ""),
                    "external_path": j.get("externalPath", ""),
                    "employer_key": employer_key,
                    "employer_name": employer["name"],
                    "location_decision": triage.decision,
                    "location_reason": triage.reason,
                }
            )

        offset += page_size
        page_num = offset // page_size
        if offset >= total:
            break
        if page_num >= max_pages:
            log.info("%s: capped at %d pages (%d results scanned)", employer["name"], max_pages, offset)
            break
        if max_results and len(all_jobs) >= max_results:
            all_jobs = all_jobs[:max_results]
            break

    log.info("%s: %d jobs found%s", employer["name"], len(all_jobs), " (filtered)" if location_filter else "")
    return all_jobs


# -- Fetch details -----------------------------------------------------------


def _fetch_one_detail(
    employer: dict,
    job: dict,
    *,
    location_filter: bool = True,
    accept_locs: list[str] | None = None,
    reject_locs: list[str] | None = None,
    location_policy: str = "recall_first",
    search_cfg: dict | None = None,
) -> dict:
    """Fetch detail for a single job."""
    try:
        detail = workday_detail(employer, job["external_path"])
        info = detail.get("jobPostingInfo", {})

        raw_desc = info.get("jobDescription", "")
        job["full_description"] = strip_html(raw_desc)
        job["apply_url"] = info.get("externalUrl", "")
        job["job_req_id"] = info.get("jobReqId", "")
        job["time_type"] = info.get("timeType", "")
        job["remote_type"] = info.get("remoteType", "")
        job["detail_location"] = _detail_location_text(info)

        if location_filter and accept_locs is not None and reject_locs is not None:
            triage = _triage_job_location(
                job,
                accept_locs,
                reject_locs,
                policy=location_policy,
                search_cfg=search_cfg,
            )
            job["location_decision"] = triage.decision
            job["location_reason"] = triage.reason

    except Exception as e:
        job["full_description"] = ""
        job["apply_url"] = ""
        job["detail_error"] = str(e)

    return job


def fetch_details(
    employer: dict,
    jobs: list[dict],
    *,
    location_filter: bool = True,
    accept_locs: list[str] | None = None,
    reject_locs: list[str] | None = None,
    location_policy: str = "recall_first",
    search_cfg: dict | None = None,
) -> list[dict]:
    """Fetch full description + apply URL for each job sequentially."""
    log.info("%s: fetching details for %d jobs...", employer["name"], len(jobs))

    kept: list[dict] = []
    completed = 0
    errors = 0
    t0 = time.time()

    for job in jobs:
        _fetch_one_detail(
            employer,
            job,
            location_filter=location_filter,
            accept_locs=accept_locs,
            reject_locs=reject_locs,
            location_policy=location_policy,
            search_cfg=search_cfg,
        )
        completed += 1
        if "detail_error" in job:
            errors += 1
        if job.get("location_decision") != "reject":
            kept.append(job)

        if completed % 20 == 0 or completed == len(jobs):
            elapsed = time.time() - t0
            rate = completed / elapsed if elapsed > 0 else 0
            log.info("%s: %d/%d (%d errors) [%.1f jobs/sec]", employer["name"], completed, len(jobs), errors, rate)

    elapsed = time.time() - t0
    log.info(
        "%s: done in %.1fs (%.1f jobs/sec, %d kept after detail triage)",
        employer["name"],
        elapsed,
        len(jobs) / elapsed if elapsed > 0 else 0,
        len(kept),
    )
    return kept


def _triage_job_location(
    job: dict,
    accept_locs: list[str],
    reject_locs: list[str],
    *,
    policy: str = "recall_first",
    search_cfg: dict | None = None,
) -> LocationTriage:
    return _triage_location(_job_location_text(job), accept_locs, reject_locs, policy=policy, search_cfg=search_cfg)


def _job_location_text(job: dict) -> str:
    parts = [
        job.get("location", ""),
        job.get("detail_location", ""),
        job.get("remote_type", ""),
        job.get("time_type", ""),
    ]
    return " | ".join(str(part) for part in parts if part)


def _detail_location_text(info: dict) -> str:
    """Extract structured location-ish fields from Workday detail JSON."""
    values: list[str] = []
    for key, value in info.items():
        lower_key = str(key).casefold()
        if "location" in lower_key and key != "jobDescription":
            values.extend(_flatten_location_value(value))
    return " | ".join(values)


def _flatten_location_value(value) -> list[str]:
    if value is None:
        return []
    if isinstance(value, str):
        return [value]
    if isinstance(value, dict):
        return [
            str(v)
            for k, v in value.items()
            if isinstance(v, str) and k in {"descriptor", "name", "city", "country", "formattedLocation"}
        ]
    if isinstance(value, list):
        flattened: list[str] = []
        for item in value:
            flattened.extend(_flatten_location_value(item))
        return flattened
    return []


# -- DB storage --------------------------------------------------------------


def store_results(conn: sqlite3.Connection, jobs: list[dict], employers: dict) -> tuple[int, int]:
    """Store corporate jobs in DB. Returns (new, existing)."""
    now = datetime.now(timezone.utc).isoformat()
    new = 0
    existing = 0

    for job in jobs:
        url = job.get("apply_url", "")
        if not url:
            emp = employers.get(job.get("employer_key", ""), {})
            if emp and job.get("external_path"):
                url = f"{emp['base_url']}/{emp['site_id']}{job['external_path']}"
        if not url:
            continue

        description = job.get("full_description", "")
        short_desc = description[:500] if description else None
        full_description = description if len(description) > 200 else None
        detail_scraped_at = now if full_description else None
        detail_error = job.get("detail_error")

        site = job.get("employer_name", "Corporate")
        strategy = "workday_api"
        review_status = REVIEW_STATUS_MANUAL if job.get("location_decision") == "manual_review" else None
        review_reason = f"workday_location:{job.get('location_reason', 'manual_review')}" if review_status else None

        try:
            conn.execute(
                "INSERT INTO jobs (url, title, salary, description, location, site, strategy, "
                "discovered_at, full_description, application_url, detail_scraped_at, detail_error, "
                "review_status, review_reason) "
                "VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?)",
                (
                    url,
                    job.get("title"),
                    None,
                    short_desc,
                    job.get("location"),
                    site,
                    strategy,
                    now,
                    full_description,
                    url,
                    detail_scraped_at,
                    detail_error,
                    review_status,
                    review_reason,
                ),
            )
            new += 1
        except sqlite3.IntegrityError:
            existing += 1

    conn.commit()
    return new, existing


def _process_one(
    employer_key: str,
    employers: dict,
    search_text: str,
    location_filter: bool,
    accept_locs: list[str],
    reject_locs: list[str],
    location_policy: str,
    search_cfg: dict | None = None,
) -> dict:
    """Search one employer, fetch details, store results."""
    emp = employers[employer_key]

    try:
        jobs = search_employer(
            employer_key,
            emp,
            search_text,
            location_filter=location_filter,
            accept_locs=accept_locs,
            reject_locs=reject_locs,
            location_policy=location_policy,
            search_cfg=search_cfg,
        )
    except Exception as e:
        log.error("%s: ERROR searching '%s': %s", emp["name"], search_text, e)
        emit_error(
            "workday_employer_search_failed",
            e,
            stage="discover",
            source="workday",
            employer_key=employer_key,
            employer=emp["name"],
            query=search_text,
        )
        return {"employer": emp["name"], "query": search_text, "found": 0, "new": 0, "existing": 0, "error": str(e)}

    if not jobs:
        return {"employer": emp["name"], "query": search_text, "found": 0, "new": 0, "existing": 0}

    try:
        jobs = fetch_details(
            emp,
            jobs,
            location_filter=location_filter,
            accept_locs=accept_locs,
            reject_locs=reject_locs,
            location_policy=location_policy,
            search_cfg=search_cfg,
        )
    except Exception as e:
        log.error("%s: ERROR fetching details for '%s': %s", emp["name"], search_text, e)
        emit_error(
            "workday_employer_detail_failed",
            e,
            stage="discover",
            source="workday",
            employer_key=employer_key,
            employer=emp["name"],
            query=search_text,
            found_before_detail=len(jobs),
        )

    conn = get_connection()
    new, existing = store_results(conn, jobs, employers)
    log.info("%s: %d new, %d already in DB", emp["name"], new, existing)
    emit_event(
        "workday_employer_finished",
        stage="discover",
        source="workday",
        status="ok",
        employer_key=employer_key,
        employer=emp["name"],
        query=search_text,
        found=len(jobs),
        new=new,
        existing=existing,
    )

    return {"employer": emp["name"], "query": search_text, "found": len(jobs), "new": new, "existing": existing}


# -- Main orchestrator -------------------------------------------------------


def scrape_employers(
    search_text: str,
    employers: dict,
    employer_keys: list[str] | None = None,
    location_filter: bool = True,
    max_results: int = 0,
    accept_locs: list[str] | None = None,
    reject_locs: list[str] | None = None,
    location_policy: str = "recall_first",
    search_cfg: dict | None = None,
    workers: int = 1,
) -> dict:
    """Run full scrape: search -> filter -> detail -> store.

    Sequential by default. When workers > 1, processes employers in parallel
    using ThreadPoolExecutor.
    """
    if employer_keys is None:
        employer_keys = list(employers.keys())

    if accept_locs is None:
        accept_locs = []
    if reject_locs is None:
        reject_locs = []

    # Ensure DB schema
    init_db()

    total_new = 0
    total_existing = 0
    total_found = 0
    errors = 0
    t0 = time.time()

    valid_keys = [k for k in employer_keys if k in employers]

    if workers > 1 and len(valid_keys) > 1:
        # Parallel mode
        completed = 0
        with ThreadPoolExecutor(max_workers=min(workers, len(valid_keys))) as pool:
            futures = {
                pool.submit(
                    _process_one,
                    key,
                    employers,
                    search_text,
                    location_filter,
                    accept_locs,
                    reject_locs,
                    location_policy,
                    search_cfg,
                ): key
                for key in valid_keys
            }
            for future in as_completed(futures):
                result = future.result()
                completed += 1
                total_new += result["new"]
                total_existing += result["existing"]
                total_found += result["found"]
                if "error" in result:
                    errors += 1

                if completed % 10 == 0 or completed == len(valid_keys):
                    elapsed = time.time() - t0
                    log.info(
                        "[%s] Progress: %d/%d employers (%d new, %d dupes, %d errors) [%.0fs]",
                        search_text,
                        completed,
                        len(valid_keys),
                        total_new,
                        total_existing,
                        errors,
                        elapsed,
                    )
    else:
        # Sequential mode (default)
        completed = 0
        for key in valid_keys:
            result = _process_one(
                key,
                employers,
                search_text,
                location_filter,
                accept_locs,
                reject_locs,
                location_policy,
                search_cfg,
            )
            completed += 1
            total_new += result["new"]
            total_existing += result["existing"]
            total_found += result["found"]
            if "error" in result:
                errors += 1

            if completed % 10 == 0 or completed == len(valid_keys):
                elapsed = time.time() - t0
                log.info(
                    "[%s] Progress: %d/%d employers (%d new, %d dupes, %d errors) [%.0fs]",
                    search_text,
                    completed,
                    len(valid_keys),
                    total_new,
                    total_existing,
                    errors,
                    elapsed,
                )

    elapsed = time.time() - t0
    log.info(
        "[%s] Done: %d found, %d new, %d dupes in %.0fs", search_text, total_found, total_new, total_existing, elapsed
    )

    return {"found": total_found, "new": total_new, "existing": total_existing}


# -- Public entry point ------------------------------------------------------


def run_workday_discovery(employers: dict | None = None, workers: int = 1) -> dict:
    """Main entry point for Workday-based corporate job discovery.

    Loads employer registry from config/employers.yaml (or uses the provided
    dict), then loads search queries from the user's search config to run
    a full crawl across all employers.

    Args:
        employers: Override the employer registry. If None, loads from YAML.
        workers: Number of parallel threads for employer scraping. Default 1 (sequential).

    Returns:
        Dict with stats: found, new, existing, queries.
    """
    search_cfg = config.load_search_config()
    if not search_cfg.get("workday_enabled", True):
        log.info("Workday discovery disabled by searches.yaml")
        return {"found": 0, "new": 0, "existing": 0, "queries": 0, "skipped": True}

    if employers is None:
        employers = load_employers()
    else:
        employers = _validate_employers(employers, source="provided")

    requested_employers = search_cfg.get("workday_employers") or []
    allow_broad_crawl = search_cfg.get("workday_allow_broad_crawl", False)
    if requested_employers:
        employers = _select_employers(employers, requested_employers)
    elif not allow_broad_crawl:
        log.info("Workday discovery skipped: no workday_employers configured and broad crawl is disabled")
        return {"found": 0, "new": 0, "existing": 0, "queries": 0, "skipped": True}

    if not employers:
        log.warning("No Workday employers configured or selected.")
        return {"found": 0, "new": 0, "existing": 0, "queries": 0}

    queries_cfg = search_cfg.get("queries", [])
    accept_locs, reject_locs = _load_location_filter(search_cfg)

    # Default to tier 1-2 queries for workday scraping
    max_tier = search_cfg.get("workday_max_tier", 2)
    queries = [q["query"] for q in queries_cfg if q.get("tier", 99) <= max_tier]

    if not queries:
        # Fallback: use all queries
        queries = [q["query"] for q in queries_cfg]

    if not queries:
        log.warning("No search queries configured in searches.yaml.")
        return {"found": 0, "new": 0, "existing": 0, "queries": 0}

    proxy = search_cfg.get("proxy")
    if proxy:
        setup_proxy(proxy)

    location_filter = search_cfg.get("workday_location_filter", True)
    location_policy = search_cfg.get("workday_location_policy", "recall_first")

    log.info("Workday crawl: %d queries x %d employers (workers=%d)", len(queries), len(employers), workers)

    grand_new = 0
    grand_existing = 0
    grand_found = 0

    for i, query in enumerate(queries, 1):
        log.info('Query %d/%d: "%s"', i, len(queries), query)
        result = scrape_employers(
            search_text=query,
            employers=employers,
            location_filter=location_filter,
            accept_locs=accept_locs,
            reject_locs=reject_locs,
            location_policy=location_policy,
            search_cfg=search_cfg,
            workers=workers,
        )
        grand_new += result["new"]
        grand_existing += result["existing"]
        grand_found += result["found"]

    log.info(
        "Workday crawl done: %d found, %d new, %d existing across %d queries x %d employers",
        grand_found,
        grand_new,
        grand_existing,
        len(queries),
        len(employers),
    )

    return {
        "found": grand_found,
        "new": grand_new,
        "existing": grand_existing,
        "queries": len(queries),
    }


def _select_employers(employers: dict, requested: list[str]) -> dict:
    """Filter employer registry by configured keys or display names."""
    if isinstance(requested, str):
        requested = [requested]
    selected = {}
    missing = []
    for item in requested:
        wanted = _normalise_name(item)
        match = None
        for key, employer in employers.items():
            if wanted in {_normalise_name(key), _normalise_name(employer.get("name", ""))}:
                match = key
                break
        if match:
            selected[match] = employers[match]
        else:
            missing.append(str(item))
    if missing:
        log.warning("Configured Workday employers not found: %s", ", ".join(missing))
    return selected


def _normalise_name(value: object) -> str:
    return re.sub(r"[^a-z0-9]+", "", str(value).casefold())
