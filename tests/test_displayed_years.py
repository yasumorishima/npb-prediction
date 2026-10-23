"""config.displayed_years: the app keeps the current predictions until the next
season's are published (DATA_END_YEAR rolls over on 11/1, annual_update runs on 3/1)."""
import config


def _set(monkeypatch, end):
    monkeypatch.delenv("NPB_DATA_END_YEAR", raising=False)
    monkeypatch.setattr(config, "DATA_END_YEAR", end)
    monkeypatch.setattr(config, "TARGET_YEAR", end + 1)


def _exists(table):
    return lambda name: table.get(name, False)


def test_falls_back_while_next_season_is_unpublished(monkeypatch):
    _set(monkeypatch, 2026)  # November 2026: config already points at 2027
    ex = _exists({"marcel_hitters_2026.csv": True})
    assert config.displayed_years(ex) == (2025, 2026)


def test_switches_once_next_season_is_published(monkeypatch):
    _set(monkeypatch, 2026)
    ex = _exists({"marcel_hitters_2026.csv": True, "marcel_hitters_2027.csv": True})
    assert config.displayed_years(ex) == (2026, 2027)


def test_unknown_keeps_configured_years(monkeypatch):
    _set(monkeypatch, 2026)
    assert config.displayed_years(lambda name: None) == (2026, 2027)
    ex = {"marcel_hitters_2027.csv": False, "marcel_hitters_2026.csv": None}
    assert config.displayed_years(ex.get) == (2026, 2027)
    # a network error on the new season's file must not switch back a season
    ex = {"marcel_hitters_2027.csv": None, "marcel_hitters_2026.csv": True}
    assert config.displayed_years(ex.get) == (2026, 2027)


def test_no_fallback_when_previous_is_missing_too(monkeypatch):
    _set(monkeypatch, 2026)
    assert config.displayed_years(_exists({})) == (2026, 2027)


def test_explicit_env_is_respected(monkeypatch):
    monkeypatch.setenv("NPB_DATA_END_YEAR", "2026")
    monkeypatch.setattr(config, "DATA_END_YEAR", 2026)
    monkeypatch.setattr(config, "TARGET_YEAR", 2027)
    ex = _exists({"marcel_hitters_2026.csv": True})
    assert config.displayed_years(ex) == (2026, 2027)
