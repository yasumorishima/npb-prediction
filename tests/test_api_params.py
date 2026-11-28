"""api.py: imports with current fastapi (Query(enum=...) is rejected from 0.160),
keeps the allowed values in the OpenAPI schema and answers 422 for an unknown sort key."""
import sys
from pathlib import Path

import pytest

pytest.importorskip("fastapi")
pytest.importorskip("httpx")
sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
from fastapi.testclient import TestClient  # noqa: E402

import api  # noqa: E402

cl = TestClient(api.app)


def test_schema_keeps_the_allowed_values():
    sch = api.app.openapi()["paths"]
    prm = {p["name"]: p["schema"] for p in sch["/rankings/hitters"]["get"]["parameters"]}
    assert prm["sort_by"]["enum"] == ["OPS", "AVG", "HR", "RBI"]
    prm = {p["name"]: p["schema"] for p in sch["/rankings/pitchers"]["get"]["parameters"]}
    assert prm["sort_by"]["enum"] == ["ERA", "WHIP", "SO", "W"]
    prm = {p["name"]: p["schema"] for p in sch["/standings/simulation"]["get"]["parameters"]}
    assert prm["league"]["enum"] == ["CL", "PL"]


@pytest.mark.parametrize("path, good, bad", [("/rankings/hitters", "HR", "XYZ"), ("/rankings/pitchers", "W", "OPS")])
def test_sort_keys(path, good, bad):
    r = cl.get(path, params={"sort_by": good, "top": 3})
    assert r.status_code in (200, 503)            # 503 only if the CSV is not in the checkout
    assert cl.get(path, params={"sort_by": bad}).status_code == 422
