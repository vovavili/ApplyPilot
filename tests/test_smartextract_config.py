from applypilot.discovery.smartextract import build_scrape_targets, select_configured_sites


def test_select_configured_sites_matches_names_case_insensitively():
    sites = [
        {"name": "RemoteOK", "url": "https://remoteok.com", "type": "static"},
        {"name": "WelcomeToTheJungle", "url": "https://example.com?q={query_encoded}", "type": "search"},
        {"name": "Job Bank Canada", "url": "https://example.ca", "type": "search"},
    ]

    selected = select_configured_sites(
        sites,
        {"smart_extract_sites": ["remote ok", "welcometothejungle"]},
    )

    assert [site["name"] for site in selected] == ["RemoteOK", "WelcomeToTheJungle"]


def test_build_scrape_targets_uses_selected_smart_extract_sites():
    sites = [
        {"name": "RemoteOK", "url": "https://remoteok.com", "type": "static"},
        {"name": "WelcomeToTheJungle", "url": "https://example.com?q={query_encoded}", "type": "search"},
    ]
    search_cfg = {"queries": [{"query": "Data Engineer"}], "locations": [{"location": "Netherlands"}]}

    targets = build_scrape_targets(sites=sites, search_cfg=search_cfg)

    assert targets == [
        {"name": "RemoteOK", "url": "https://remoteok.com", "query": None},
        {"name": "WelcomeToTheJungle", "url": "https://example.com?q=Data+Engineer", "query": "Data Engineer"},
    ]
