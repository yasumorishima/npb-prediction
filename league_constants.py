"""
League constants for the Streamlit app (aggregates only, no per-player rows).

The app used to download the full per-player sabermetrics table
(data/projections/npb_sabermetrics_2015_<END>.csv) only to derive two league
aggregates from it. That table is a copy of baseball-data.com / npb.jp stat
lines; neither site grants a redistribution right, so it is kept in the private
repo yasumorishima/npb-restricted-inputs. This script computes the aggregates
from the restricted table and writes data/projections/league_constants.json,
which is public.

Computed exactly as streamlit_app.py did from the table:
  woba_fit : least squares  wOBA ~ a*OBP + b*SLG + c  on rows with PA >= 100
             and non-null wOBA / OBP / SLG
             (_enrich_projections / _ensure_hitter_saber / _build_2026_standings)
  lg_woba  : mean wOBA over rows with year >= 2022 and PA >= 50
             (_enrich_projections / _ensure_hitter_saber / _build_2026_standings)

The app also loaded data/raw/npb_pitchers_2015_<END>.csv, but only for the
per-player ERA/FIP history chart, which was removed; it derived no league
aggregate from that table that the remaining pages use.

Usage (repo root, restricted table present):
    python league_constants.py            # uses config.DATA_END_YEAR
    NPB_DATA_END_YEAR=2025 python league_constants.py

Data sources: プロ野球データFreak (https://baseball-data.com) / 日本野球機構 NPB (https://npb.jp)
"""

import json
import math
import sys
from pathlib import Path

import numpy as np
import pandas as pd

from config import DATA_END_YEAR

ROOT = Path(__file__).parent
SOURCE_REL = f"data/projections/npb_sabermetrics_2015_{DATA_END_YEAR}.csv"
OUT_PATH = ROOT / "data" / "projections" / "league_constants.json"

FIT_MIN_PA = 100
LG_WOBA_MIN_YEAR = 2022
LG_WOBA_MIN_PA = 50


def _f(x) -> float | None:
    """float for JSON; NaN -> None (the app turns None back into NaN)."""
    x = float(x)
    return None if math.isnan(x) else x


def compute(saber: pd.DataFrame) -> dict:
    """Return the aggregates the app needs. Mirrors the app's code line by line."""
    df_fit = saber[saber["PA"] >= FIT_MIN_PA].dropna(subset=["wOBA", "OBP", "SLG"])
    X = np.column_stack([df_fit["OBP"].values, df_fit["SLG"].values, np.ones(len(df_fit))])
    coeffs, _, _, _ = np.linalg.lstsq(X, df_fit["wOBA"].values, rcond=None)
    a_obp, b_slg, intercept_w = coeffs

    recent_s = saber[saber["year"] >= LG_WOBA_MIN_YEAR]
    lg_woba = recent_s[recent_s["PA"] >= LG_WOBA_MIN_PA]["wOBA"].mean()

    return {
        "woba_fit": {
            "a_obp": _f(a_obp),
            "b_slg": _f(b_slg),
            "intercept": _f(intercept_w),
            "n_rows": int(len(df_fit)),
            "rule": f"lstsq wOBA ~ OBP + SLG + 1 on PA >= {FIT_MIN_PA}, non-null wOBA/OBP/SLG",
        },
        "lg_woba": {
            "value": _f(lg_woba),
            "n_rows_recent": int(len(recent_s)),
            "n_rows_used": int((recent_s["PA"] >= LG_WOBA_MIN_PA).sum()),
            "rule": f"mean wOBA over year >= {LG_WOBA_MIN_YEAR} and PA >= {LG_WOBA_MIN_PA}",
        },
    }


def main() -> int:
    src = ROOT / SOURCE_REL
    if not src.exists():
        print(f"ERROR: {SOURCE_REL} not found (restricted table; see DATA_SOURCES.md)", file=sys.stderr)
        return 1
    # Same read as streamlit_app.load_csv (utf-8-sig); player/team name
    # normalisation does not touch the numeric columns used here.
    saber = pd.read_csv(src, encoding="utf-8-sig")
    out = {
        "source_table": Path(SOURCE_REL).name,
        "data_end_year": int(DATA_END_YEAR),
        "year_min": int(saber["year"].min()),
        "year_max": int(saber["year"].max()),
        **compute(saber),
        "credits": "baseball-data.com (プロ野球データFreak) / npb.jp (日本野球機構)",
    }
    OUT_PATH.parent.mkdir(parents=True, exist_ok=True)
    with open(OUT_PATH, "w", encoding="utf-8") as f:
        json.dump(out, f, ensure_ascii=False, indent=2, allow_nan=False)
        f.write("\n")
    print(f"Saved: {OUT_PATH.relative_to(ROOT)}")
    print(json.dumps(out, ensure_ascii=False, indent=2))
    return 0


if __name__ == "__main__":
    sys.exit(main())
