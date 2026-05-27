from __future__ import annotations

import json
import urllib.error

from hypothesis import given
from hypothesis import strategies as st

from applypilot.discovery import location


def _train_cfg(**overrides) -> dict:
    source = {
        "static_table": True,
        "ns_api_fallback": False,
        "cache_path": "",
        "max_api_lookups_per_run": 10,
        "min_seconds_between_requests": 0,
    }
    source.update(overrides.pop("source", {}))
    policy = {
        "enabled": True,
        "max_minutes": 100,
        "unknown_city": "manual_review",
        "over_max_minutes": "manual_review",
        "anchors": [
            {"station": "Den Haag Centraal", "code": "GVC"},
            {"station": "Rotterdam Centraal", "code": "RTD"},
        ],
        "source": source,
    }
    policy.update(overrides)
    return {"location_train_policy": policy}


def _triage(text: str | None, cfg: dict | None = None, reject: list[str] | None = None) -> location.LocationTriage:
    return location.triage_location(
        text,
        ["Remote", "Europe", "EU", "EMEA"],
        reject or ["United States", "USA", "Canada", "India", "Maastricht", "Limburg"],
        search_cfg=cfg or _train_cfg(),
    )


def test_parse_city_handles_real_job_location_shapes() -> None:
    assert location.parse_city("Amsterdam, North Holland, Netherlands").city_key == "amsterdam"
    assert location.parse_city("Breda, North Brabant, Netherlands").city_key == "breda"
    assert location.parse_city("North Brabant, Netherlands").reason == "province_only_location"

    for text in ("", "2 Locations", "Multiple Locations", "Global", "Hybrid"):
        assert location.parse_city(text).city_key is None


def test_static_train_policy_accepts_known_commutable_dutch_cities() -> None:
    for city in (
        "Amsterdam, North Holland, Netherlands",
        "Rotterdam, South Holland, Netherlands",
        "The Hague, South Holland, Netherlands",
        "Utrecht, Utrecht, Netherlands",
        "Delft, South Holland, Netherlands",
        "Leiden, South Holland, Netherlands",
        "Rijswijk, South Holland, Netherlands",
        "Gouda, South Holland, Netherlands",
        "Haarlem, North Holland, Netherlands",
        "Hoofddorp, North Holland, Netherlands",
        "Amersfoort, Utrecht, Netherlands",
        "Breda, North Brabant, Netherlands",
        "Schiphol, North Holland, Netherlands",
    ):
        triage = _triage(city)

        assert triage.decision == "accept"
        assert triage.reason.startswith("accepted_train_")


def test_train_policy_blocks_or_reviews_known_problem_locations() -> None:
    assert _triage("Maastricht, Limburg, Netherlands").decision == "reject"

    for text in (
        "Enkhuizen, North Holland, Netherlands",
        "Zaltbommel, Gelderland, Netherlands",
        "Renswoude, Utrecht, Netherlands",
        "North Brabant, Netherlands",
        "Mysterydam, Netherlands",
    ):
        triage = _triage(text, reject=["United States", "USA", "Canada", "India", "Maastricht", "Limburg"])

        assert triage.decision == "manual_review"


def test_remote_and_europe_rules_still_win_before_train_checks() -> None:
    assert _triage("Remote - EMEA").decision == "accept"
    assert _triage("Europe").decision == "accept"
    assert _triage("European Union").decision == "accept"
    assert _triage("Remote-Canada-British Columbia").decision == "reject"


def test_train_threshold_boundaries(monkeypatch) -> None:
    monkeypatch.setitem(
        location.STATIC_TRAIN_COMMUTES,
        "boundaryville",
        location.StaticCommute(
            "Boundaryville",
            "Boundaryville",
            "BDV",
            {"GVC": 101, "RTD": 999},
        ),
    )

    assert _triage("Boundaryville, Netherlands", _train_cfg(max_minutes=99)).decision == "manual_review"
    assert _triage("Boundaryville, Netherlands", _train_cfg(max_minutes=100)).decision == "manual_review"
    assert _triage("Boundaryville, Netherlands", _train_cfg(max_minutes=101)).decision == "accept"

    monkeypatch.setitem(
        location.STATIC_TRAIN_COMMUTES,
        "exactville",
        location.StaticCommute("Exactville", "Exactville", "EXV", {"GVC": 100}),
    )

    assert _triage("Exactville, Netherlands", _train_cfg(max_minutes=100)).decision == "accept"


@given(
    lower=st.integers(min_value=0, max_value=160),
    delta=st.integers(min_value=0, max_value=80),
    minutes=st.integers(min_value=0, max_value=200),
)
def test_raising_threshold_cannot_make_static_city_less_accepted(lower: int, delta: int, minutes: int) -> None:
    original = location.STATIC_TRAIN_COMMUTES.get("propertyville")
    location.STATIC_TRAIN_COMMUTES["propertyville"] = location.StaticCommute(
        "Propertyville",
        "Propertyville",
        "PRV",
        {"GVC": minutes},
    )
    try:
        stricter = _triage("Propertyville, Netherlands", _train_cfg(max_minutes=lower))
        looser = _triage("Propertyville, Netherlands", _train_cfg(max_minutes=lower + delta))

        if stricter.decision == "accept":
            assert looser.decision == "accept"
    finally:
        if original is None:
            location.STATIC_TRAIN_COMMUTES.pop("propertyville", None)
        else:
            location.STATIC_TRAIN_COMMUTES["propertyville"] = original


def test_ns_cache_hit_avoids_network(monkeypatch, tmp_path) -> None:
    cache_path = tmp_path / "commutes.json"
    cache_path.write_text(
        json.dumps(
            {
                "stations": {"unknowncity": {"code": "UNK", "name": "Unknowncity"}},
                "trips": {"GVC:UNK": {"minutes": 88}, "RTD:UNK": {"minutes": 92}},
            }
        ),
        encoding="utf-8",
    )
    monkeypatch.setenv("NS_API_KEY", "test-key")
    monkeypatch.setattr(location, "_ns_get_json", lambda *_args, **_kwargs: (_ for _ in ()).throw(AssertionError))
    location.reset_ns_runtime_state()

    triage = _triage(
        "Unknowncity, Netherlands",
        _train_cfg(source={"static_table": False, "ns_api_fallback": True, "cache_path": str(cache_path)}),
    )

    assert triage.decision == "accept"
    assert triage.reason == "accepted_train_88m:den_haag"


def test_ns_fallback_is_skipped_without_api_key(monkeypatch, tmp_path) -> None:
    monkeypatch.delenv("NS_API_KEY", raising=False)
    monkeypatch.setattr(location, "_ns_get_json", lambda *_args, **_kwargs: (_ for _ in ()).throw(AssertionError))
    location.reset_ns_runtime_state()

    triage = _triage(
        "Unknowncity, Netherlands",
        _train_cfg(
            source={
                "static_table": False,
                "ns_api_fallback": True,
                "cache_path": str(tmp_path / "commutes.json"),
            }
        ),
    )

    assert triage.decision == "manual_review"
    assert triage.reason == "unknown_city"


def test_ns_success_writes_cache(monkeypatch, tmp_path) -> None:
    cache_path = tmp_path / "commutes.json"
    calls: list[tuple[str, dict[str, str]]] = []

    def fake_ns(path: str, params: dict[str, str], _api_key: str):
        calls.append((path, params))
        if path == "/v2/stations":
            return [{"code": "UNK", "name": "Unknowncity"}]
        return {"trips": [{"plannedDurationInMinutes": 73}]}

    monkeypatch.setenv("NS_API_KEY", "test-key")
    monkeypatch.setattr(location, "_ns_get_json", fake_ns)
    location.reset_ns_runtime_state()

    triage = _triage(
        "Unknowncity, Netherlands",
        _train_cfg(source={"static_table": False, "ns_api_fallback": True, "cache_path": str(cache_path)}),
    )

    assert triage.decision == "accept"
    assert triage.reason == "accepted_train_73m:den_haag"
    cache = json.loads(cache_path.read_text(encoding="utf-8"))
    assert cache["stations"]["unknowncity"]["code"] == "UNK"
    assert cache["trips"]["GVC:UNK"]["minutes"] == 73
    assert [call[0] for call in calls] == ["/v2/stations", "/v3/trips", "/v3/trips"]


def test_ns_failure_modes_return_manual_review(monkeypatch, tmp_path) -> None:
    failures = [
        urllib.error.HTTPError("https://ns.example", 401, "Unauthorized", None, None),
        urllib.error.HTTPError("https://ns.example", 403, "Forbidden", None, None),
        urllib.error.HTTPError("https://ns.example", 429, "Too Many Requests", None, None),
        TimeoutError("timeout"),
        ValueError("bad json"),
    ]

    for index, exc in enumerate(failures):
        monkeypatch.setenv("NS_API_KEY", "test-key")
        monkeypatch.setattr(location, "_ns_get_json", lambda *_args, _exc=exc, **_kwargs: (_ for _ in ()).throw(_exc))
        location.reset_ns_runtime_state()

        triage = _triage(
            f"Unknowncity{index}, Netherlands",
            _train_cfg(
                source={
                    "static_table": False,
                    "ns_api_fallback": True,
                    "cache_path": str(tmp_path / f"commutes-{index}.json"),
                }
            ),
        )

        assert triage.decision == "manual_review"
        assert triage.reason == "unknown_city"


def test_ns_ambiguous_station_or_no_route_returns_manual_review(monkeypatch, tmp_path) -> None:
    def ambiguous_station(path: str, _params: dict[str, str], _api_key: str):
        if path == "/v2/stations":
            return [{"code": "AA", "name": "Unknowncity Centraal"}, {"code": "AB", "name": "Unknowncity West"}]
        return {"trips": [{"plannedDurationInMinutes": 80}]}

    monkeypatch.setenv("NS_API_KEY", "test-key")
    monkeypatch.setattr(location, "_ns_get_json", ambiguous_station)
    location.reset_ns_runtime_state()

    assert (
        _triage(
            "Unknowncity, Netherlands",
            _train_cfg(source={"static_table": False, "ns_api_fallback": True, "cache_path": str(tmp_path / "a.json")}),
        ).decision
        == "manual_review"
    )

    def no_route(path: str, _params: dict[str, str], _api_key: str):
        if path == "/v2/stations":
            return [{"code": "UNK", "name": "Unknowncity"}]
        return {"trips": []}

    monkeypatch.setattr(location, "_ns_get_json", no_route)
    location.reset_ns_runtime_state()

    assert (
        _triage(
            "Unknowncity, Netherlands",
            _train_cfg(source={"static_table": False, "ns_api_fallback": True, "cache_path": str(tmp_path / "b.json")}),
        ).decision
        == "manual_review"
    )
