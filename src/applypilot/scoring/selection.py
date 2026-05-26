"""Shared job selection rules for scoring."""

from applypilot import config


def score_selection_sql(*, rescore: bool = False) -> tuple[list[str], list[object]]:
    """Return SQL filters for jobs that should be scored.

    This keeps the scorer, pipeline progress checks, and status output aligned.
    """
    search_cfg = config.load_search_config()
    where = ["full_description IS NOT NULL"]
    params: list[object] = []

    if not rescore:
        where.append("fit_score IS NULL")

    where.append("COALESCE(review_status, '') != 'manual_review'")

    if not search_cfg.get("workday_enabled", True) and not search_cfg.get("score_workday_when_disabled", False):
        where.append("COALESCE(strategy, '') != 'workday_api'")

    for title in search_cfg.get("exclude_titles", []):
        title = str(title).strip().lower()
        if title:
            where.append("LOWER(COALESCE(title, '')) NOT LIKE ?")
            params.append(f"%{title}%")

    return where, params


def count_jobs_to_score(conn, *, rescore: bool = False) -> int:
    """Count jobs matching the same filters used by the scorer."""
    where, params = score_selection_sql(rescore=rescore)
    return conn.execute(
        f"SELECT COUNT(*) FROM jobs WHERE {' AND '.join(where)}",
        params,
    ).fetchone()[0]
