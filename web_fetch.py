"""Fetch an HTML page with an honest User-Agent and parse its tables.

baseball-data.com answers 403 to Python urllib's default User-Agent (measured
2026-11-28 from cloud; on 2026-10-23 it still answered 200) and to the
python-requests default, so `pd.read_html(url)` no longer works there. This is
the User-Agent `research/team_2026/games26.py` already uses (CLAUDE.md: never a
made-up browser User-Agent). npb.jp still answers urllib's default; using this
for it as well is harmless.
"""
from __future__ import annotations

import io
import urllib.request

import pandas as pd

HEADERS = {"User-Agent": "Mozilla/5.0 (compatible; research-bot/1.0)"}


def get_bytes(url: str, timeout: float = 30) -> bytes:
    req = urllib.request.Request(url, headers=HEADERS)
    with urllib.request.urlopen(req, timeout=timeout) as resp:
        return resp.read()


def read_html_tables(url: str, encoding: str = "utf-8", **kw) -> list[pd.DataFrame]:
    """pd.read_html(url, encoding=...) with HEADERS (same parse, from the page bytes)."""
    return pd.read_html(io.BytesIO(get_bytes(url)), encoding=encoding, **kw)
