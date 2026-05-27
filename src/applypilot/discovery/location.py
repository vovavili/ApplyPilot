"""Location triage for remote, EU, and Dutch train-commute eligibility."""

from __future__ import annotations

import json
import logging
import os
import re
import time
import urllib.parse
import urllib.request
from dataclasses import dataclass
from pathlib import Path
from typing import Any

from applypilot import config

log = logging.getLogger(__name__)

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
EUROPE_TERMS = ("europe", "european union", "emea")
SHORT_EU_TERMS = ("eu",)
LEGACY_NL_TERMS = (
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
)
VAGUE_EXACT_TERMS = {"", "hybrid", "on-site", "onsite", "office"}
DUTCH_PROVINCES = {
    "drenthe",
    "flevoland",
    "friesland",
    "fryslan",
    "gelderland",
    "groningen",
    "limburg",
    "north brabant",
    "noord-brabant",
    "north holland",
    "noord-holland",
    "overijssel",
    "south holland",
    "zuid-holland",
    "utrecht",
    "zeeland",
}
DUTCH_COUNTRY_TERMS = ("netherlands", "nederland", "nl")
NS_API_BASE_URL = "https://gateway.apiportal.ns.nl/reisinformatie-api/api"


@dataclass(frozen=True)
class LocationTriage:
    decision: str
    reason: str


@dataclass(frozen=True)
class ParsedLocation:
    city: str | None
    city_key: str | None
    reason: str | None = None


@dataclass(frozen=True)
class StaticCommute:
    city: str
    station: str
    code: str
    minutes: dict[str, int]


DEFAULT_TRAIN_POLICY = {
    "enabled": False,
    "max_minutes": 100,
    "unknown_city": "manual_review",
    "over_max_minutes": "manual_review",
    "anchors": [
        {"station": "Den Haag Centraal", "code": "GVC"},
        {"station": "Rotterdam Centraal", "code": "RTD"},
    ],
    "source": {
        "static_table": True,
        "ns_api_fallback": False,
        "cache_path": str(config.APP_DIR / "train_commute_cache.json"),
        "max_api_lookups_per_run": 10,
        "min_seconds_between_requests": 1.1,
    },
}


CITY_ALIASES = {
    "'s gravenhage": "den haag",
    "'s-gravenhage": "den haag",
    "s gravenhage": "den haag",
    "s-gravenhage": "den haag",
    "the hague": "den haag",
    "den haag": "den haag",
    "schiphol airport": "schiphol",
    "amsterdam zuid": "amsterdam",
    "utrecht centraal": "utrecht",
}


STATIC_TRAIN_COMMUTES: dict[str, StaticCommute] = {
    "amersfoort": StaticCommute("Amersfoort", "Amersfoort Centraal", "AMF", {"GVC": 67, "RTD": 61}),
    "amsterdam": StaticCommute("Amsterdam", "Amsterdam Centraal", "ASD", {"GVC": 50, "RTD": 41}),
    "amstelveen": StaticCommute("Amstelveen", "Amsterdam Zuid", "ASDZ", {"GVC": 48, "RTD": 46}),
    "breda": StaticCommute("Breda", "Breda", "BD", {"GVC": 49, "RTD": 24}),
    "de meern": StaticCommute("De Meern", "Utrecht Centraal", "UT", {"GVC": 47, "RTD": 39}),
    "delft": StaticCommute("Delft", "Delft", "DT", {"GVC": 13, "RTD": 14}),
    "den haag": StaticCommute("Den Haag", "Den Haag Centraal", "GVC", {"GVC": 0, "RTD": 24}),
    "diemen": StaticCommute("Diemen", "Diemen", "DMN", {"GVC": 61, "RTD": 58}),
    "gouda": StaticCommute("Gouda", "Gouda", "GD", {"GVC": 19, "RTD": 19}),
    "haarlem": StaticCommute("Haarlem", "Haarlem", "HLM", {"GVC": 45, "RTD": 58}),
    "hilversum": StaticCommute("Hilversum", "Hilversum", "HVS", {"GVC": 66, "RTD": 63}),
    "hoofddorp": StaticCommute("Hoofddorp", "Hoofddorp", "HFD", {"GVC": 37, "RTD": 44}),
    "leiden": StaticCommute("Leiden", "Leiden Centraal", "LEDN", {"GVC": 12, "RTD": 33}),
    "rijswijk": StaticCommute("Rijswijk", "Rijswijk", "RSW", {"GVC": 5, "RTD": 18}),
    "rotterdam": StaticCommute("Rotterdam", "Rotterdam Centraal", "RTD", {"GVC": 24, "RTD": 0}),
    "schiphol": StaticCommute("Schiphol", "Schiphol Airport", "SHL", {"GVC": 30, "RTD": 26}),
    "utrecht": StaticCommute("Utrecht", "Utrecht Centraal", "UT", {"GVC": 39, "RTD": 38}),
    "zaandam": StaticCommute("Zaandam", "Zaandam", "ZD", {"GVC": 63, "RTD": 63}),
}

_ns_lookup_count = 0
_last_ns_request_at = 0.0


def triage_location(
    location: str | None,
    accept: list[str],
    reject: list[str],
    *,
    policy: str = "recall_first",
    search_cfg: dict | None = None,
) -> LocationTriage:
    text = str(location or "").strip()
    if not text:
        return _ambiguous_location(policy, "blank_location")

    loc = text.casefold()
    if _contains_any(loc, EUROPE_TERMS) or _contains_short_term(loc, SHORT_EU_TERMS):
        return LocationTriage("accept", "accepted_europe_emea")

    if _contains_any(loc, REMOTE_TERMS):
        if _contains_any(loc, reject):
            return LocationTriage("reject", "rejected_remote_restricted_foreign")
        return LocationTriage("accept", "accepted_remote")

    train_policy = _merged_train_policy(search_cfg)
    if train_policy.get("enabled"):
        return _triage_train_location(text, loc, reject, train_policy, policy)

    return _triage_legacy_location(loc, accept, reject, policy)


def parse_city(location: str | None) -> ParsedLocation:
    text = str(location or "").strip()
    if not text:
        return ParsedLocation(None, None, "blank_location")

    lowered = text.casefold().strip()
    if _is_vague_text(lowered):
        return ParsedLocation(None, None, "ambiguous_location")

    if "|" in text:
        pieces = [piece.strip() for piece in text.split("|") if piece.strip()]
        city_keys = {parse_city(piece).city_key for piece in pieces}
        city_keys.discard(None)
        if len(city_keys) != 1:
            return ParsedLocation(None, None, "ambiguous_location")
        key = city_keys.pop()
        return ParsedLocation(_display_city(key), key)

    first = re.split(r"[,;]", text, maxsplit=1)[0].strip()
    first = re.sub(r"\([^)]*\)", "", first).strip()
    first = re.sub(r"\s+", " ", first)
    if not first:
        return ParsedLocation(None, None, "blank_location")

    key = _normalise_city(first)
    if _is_vague_text(key):
        return ParsedLocation(None, None, "ambiguous_location")
    if key in DUTCH_PROVINCES and _static_commute(key) is None:
        return ParsedLocation(None, None, "province_only_location")
    return ParsedLocation(first, key)


def reset_ns_runtime_state() -> None:
    global _ns_lookup_count, _last_ns_request_at
    _ns_lookup_count = 0
    _last_ns_request_at = 0.0


def _triage_train_location(
    text: str,
    loc: str,
    reject: list[str],
    train_policy: dict,
    policy: str,
) -> LocationTriage:
    if _contains_any(loc, reject):
        return LocationTriage("reject", "rejected_non_remote_foreign")
    if _is_vague_text(loc):
        return _ambiguous_location(policy, "ambiguous_location")

    parsed = parse_city(text)
    if not parsed.city_key:
        return _ambiguous_location(policy, parsed.reason or "unknown_city")

    if not _looks_dutch_location(loc, parsed.city_key):
        return _ambiguous_location(policy, "unmatched_location")

    source = train_policy.get("source") or {}
    commute = _static_commute(parsed.city_key) if source.get("static_table", True) else None
    if commute:
        return _triage_commute_minutes(commute.minutes, train_policy, policy)

    minutes = _ns_fallback_minutes(parsed.city_key, train_policy)
    if minutes:
        return _triage_commute_minutes(minutes, train_policy, policy)

    return _policy_decision(train_policy.get("unknown_city", "manual_review"), policy, "unknown_city")


def _triage_commute_minutes(minutes_by_anchor: dict[str, int], train_policy: dict, policy: str) -> LocationTriage:
    anchors = _policy_anchors(train_policy)
    relevant = [
        (anchor, minutes_by_anchor[anchor["code"]]) for anchor in anchors if anchor["code"] in minutes_by_anchor
    ]
    if not relevant:
        return _ambiguous_location(policy, "unknown_commute")

    anchor, minutes = min(relevant, key=lambda item: item[1])
    max_minutes = int(train_policy.get("max_minutes", 100))
    anchor_label = _anchor_label(anchor)
    if minutes <= max_minutes:
        return LocationTriage("accept", f"accepted_train_{minutes}m:{anchor_label}")
    return _policy_decision(
        train_policy.get("over_max_minutes", "manual_review"),
        policy,
        f"train_over_{max_minutes}m:{minutes}m:{anchor_label}",
    )


def _triage_legacy_location(loc: str, accept: list[str], reject: list[str], policy: str) -> LocationTriage:
    if _contains_any(loc, LEGACY_NL_TERMS):
        return LocationTriage("accept", "accepted_nl_or_configured")
    if _contains_accept_term(loc, accept):
        return LocationTriage("accept", "accepted_nl_or_configured")
    if _contains_any(loc, VAGUE_LOCATION_TERMS) or loc in VAGUE_EXACT_TERMS:
        return _ambiguous_location(policy, "ambiguous_location")
    if _contains_any(loc, reject):
        return LocationTriage("reject", "rejected_non_remote_foreign")
    return _ambiguous_location(policy, "unmatched_location")


def _ns_fallback_minutes(city_key: str, train_policy: dict) -> dict[str, int] | None:
    source = train_policy.get("source") or {}
    if not source.get("ns_api_fallback"):
        return None

    api_key = os.environ.get("NS_API_KEY", "").strip()
    if not api_key:
        return None

    cache = _load_cache(Path(source.get("cache_path") or DEFAULT_TRAIN_POLICY["source"]["cache_path"]))
    station = _resolve_station(city_key, train_policy, cache, api_key)
    if not station:
        _write_cache(cache, Path(source.get("cache_path") or DEFAULT_TRAIN_POLICY["source"]["cache_path"]))
        return None

    minutes: dict[str, int] = {}
    for anchor in _policy_anchors(train_policy):
        value = _resolve_trip_minutes(anchor["code"], station["code"], train_policy, cache, api_key)
        if value is not None:
            minutes[anchor["code"]] = value

    _write_cache(cache, Path(source.get("cache_path") or DEFAULT_TRAIN_POLICY["source"]["cache_path"]))
    return minutes or None


def _resolve_station(city_key: str, train_policy: dict, cache: dict, api_key: str) -> dict | None:
    stations = cache.setdefault("stations", {})
    if city_key in stations:
        return stations[city_key]

    try:
        data = _rate_limited_ns_get_json("/v2/stations", {}, train_policy, api_key)
    except Exception as exc:
        log.warning("NS station lookup failed for %s: %s", city_key, exc)
        return None

    candidates = []
    for item in _station_items(data):
        name = _station_name(item)
        code = item.get("code") or item.get("UICCode") or item.get("uicCode")
        if not name or not code:
            continue
        key = _normalise_city(name)
        if key == city_key or key.startswith(f"{city_key} "):
            candidates.append({"code": str(code), "name": name})

    exact = [candidate for candidate in candidates if _normalise_city(candidate["name"]) == city_key]
    chosen = exact or candidates
    if len(chosen) != 1:
        return None

    stations[city_key] = chosen[0]
    return chosen[0]


def _resolve_trip_minutes(
    from_code: str,
    to_code: str,
    train_policy: dict,
    cache: dict,
    api_key: str,
) -> int | None:
    trips = cache.setdefault("trips", {})
    key = f"{from_code}:{to_code}"
    if key in trips:
        return int(trips[key]["minutes"])

    try:
        data = _rate_limited_ns_get_json(
            "/v3/trips",
            {"fromStation": from_code, "toStation": to_code, "searchForArrival": "false"},
            train_policy,
            api_key,
        )
    except Exception as exc:
        log.warning("NS trip lookup failed for %s: %s", key, exc)
        return None

    minutes = _best_trip_minutes(data)
    if minutes is None:
        return None
    trips[key] = {"minutes": minutes}
    return minutes


def _rate_limited_ns_get_json(path: str, params: dict[str, str], train_policy: dict, api_key: str) -> Any:
    global _ns_lookup_count, _last_ns_request_at
    source = train_policy.get("source") or {}
    max_lookups = int(source.get("max_api_lookups_per_run", 10))
    if _ns_lookup_count >= max_lookups:
        raise RuntimeError("NS API lookup budget exhausted")

    min_wait = float(source.get("min_seconds_between_requests", 1.1))
    elapsed = time.time() - _last_ns_request_at
    if _last_ns_request_at and elapsed < min_wait:
        time.sleep(min_wait - elapsed)

    _ns_lookup_count += 1
    _last_ns_request_at = time.time()
    return _ns_get_json(path, params, api_key)


def _ns_get_json(path: str, params: dict[str, str], api_key: str) -> Any:
    query = urllib.parse.urlencode(params)
    url = f"{NS_API_BASE_URL}{path}"
    if query:
        url = f"{url}?{query}"
    request = urllib.request.Request(
        url,
        headers={
            "Accept": "application/json",
            "Ocp-Apim-Subscription-Key": api_key,
            "User-Agent": "ApplyPilot/0.3.0 personal-job-search",
        },
    )
    with urllib.request.urlopen(request, timeout=15) as response:
        return json.loads(response.read().decode("utf-8"))


def _best_trip_minutes(data: Any) -> int | None:
    candidates = data.get("trips") if isinstance(data, dict) else data
    if isinstance(data, dict) and candidates is None:
        candidates = data.get("payload")
    if not isinstance(candidates, list):
        return None

    values = [_trip_minutes(item) for item in candidates if isinstance(item, dict)]
    values = [value for value in values if value is not None]
    return min(values) if values else None


def _trip_minutes(item: dict) -> int | None:
    for key in ("plannedDurationInMinutes", "actualDurationInMinutes", "durationInMinutes"):
        value = item.get(key)
        if isinstance(value, (int, float)):
            return int(value)
        if isinstance(value, str) and value.isdigit():
            return int(value)
    duration = item.get("duration") or item.get("plannedDuration")
    if isinstance(duration, str):
        return _parse_iso_duration_minutes(duration)
    return None


def _parse_iso_duration_minutes(value: str) -> int | None:
    match = re.fullmatch(r"P(?:T)?(?:(\d+)H)?(?:(\d+)M)?(?:(\d+)S)?", value.strip())
    if not match:
        return None
    hours = int(match.group(1) or 0)
    minutes = int(match.group(2) or 0)
    seconds = int(match.group(3) or 0)
    return hours * 60 + minutes + (1 if seconds else 0)


def _station_items(data: Any) -> list[dict]:
    if isinstance(data, list):
        return [item for item in data if isinstance(item, dict)]
    if not isinstance(data, dict):
        return []
    for key in ("payload", "stations", "data"):
        value = data.get(key)
        if isinstance(value, list):
            return [item for item in value if isinstance(item, dict)]
    return []


def _station_name(item: dict) -> str | None:
    for key in ("name", "stationName", "longName", "mediumName"):
        if item.get(key):
            return str(item[key])
    names = item.get("namen") or item.get("names")
    if isinstance(names, dict):
        for key in ("lang", "middel", "kort", "long", "medium", "short"):
            if names.get(key):
                return str(names[key])
    return None


def _load_cache(path: Path) -> dict:
    if not path.exists():
        return {"stations": {}, "trips": {}}
    try:
        data = json.loads(path.read_text(encoding="utf-8"))
    except (OSError, json.JSONDecodeError):
        return {"stations": {}, "trips": {}}
    if not isinstance(data, dict):
        return {"stations": {}, "trips": {}}
    data.setdefault("stations", {})
    data.setdefault("trips", {})
    return data


def _write_cache(cache: dict, path: Path) -> None:
    try:
        path.parent.mkdir(parents=True, exist_ok=True)
        path.write_text(json.dumps(cache, indent=2, sort_keys=True), encoding="utf-8")
    except OSError as exc:
        log.debug("Could not write train commute cache %s: %s", path, exc)


def _static_commute(city_key: str) -> StaticCommute | None:
    return STATIC_TRAIN_COMMUTES.get(CITY_ALIASES.get(city_key, city_key))


def _looks_dutch_location(loc: str, city_key: str) -> bool:
    return (
        city_key in STATIC_TRAIN_COMMUTES
        or city_key in CITY_ALIASES
        or _contains_any(loc, DUTCH_COUNTRY_TERMS)
        or _contains_any(loc, tuple(DUTCH_PROVINCES))
    )


def _merged_train_policy(search_cfg: dict | None) -> dict:
    policy = dict(DEFAULT_TRAIN_POLICY)
    if not search_cfg:
        return policy

    configured = search_cfg.get("location_train_policy") or {}
    if not isinstance(configured, dict):
        return policy

    source = {**policy["source"], **(configured.get("source") or {})}
    policy.update({key: value for key, value in configured.items() if key != "source"})
    policy["source"] = source
    return policy


def _policy_anchors(train_policy: dict) -> list[dict[str, str]]:
    anchors = train_policy.get("anchors") or DEFAULT_TRAIN_POLICY["anchors"]
    result = []
    for anchor in anchors:
        if isinstance(anchor, dict) and anchor.get("code") and anchor.get("station"):
            result.append({"station": str(anchor["station"]), "code": str(anchor["code"])})
    return result or DEFAULT_TRAIN_POLICY["anchors"]


def _anchor_label(anchor: dict[str, str]) -> str:
    station = anchor["station"].casefold().replace("centraal", "").strip()
    return re.sub(r"[^a-z0-9]+", "_", station).strip("_")


def _normalise_city(value: str) -> str:
    normalized = re.sub(r"[^a-zA-Z0-9' -]+", " ", value).casefold()
    normalized = re.sub(r"\s+", " ", normalized).strip()
    return CITY_ALIASES.get(normalized, normalized)


def _display_city(city_key: str) -> str:
    commute = STATIC_TRAIN_COMMUTES.get(city_key)
    if commute:
        return commute.city
    return city_key.title()


def _is_vague_text(text: str) -> bool:
    value = text.strip().casefold()
    if value in VAGUE_EXACT_TERMS:
        return True
    if re.fullmatch(r"\d+\s+locations?", value):
        return True
    return any(term in value for term in VAGUE_LOCATION_TERMS)


def _policy_decision(configured: str, policy: str, reason: str) -> LocationTriage:
    if str(configured).casefold() == "reject":
        return LocationTriage("reject", reason)
    return _ambiguous_location(policy, reason)


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
