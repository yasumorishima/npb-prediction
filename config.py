"""
NPB予測システム 年度設定
GitHub Actionsのenv変数 NPB_DATA_END_YEAR で上書き可能

デフォルトは「直前に終了したNPBシーズン年」:
  - 11月以降: 今年のシーズンが終了 → DATA_END_YEAR = 今年
  - 1〜10月: まだシーズン中 / 終了前 → DATA_END_YEAR = 昨年
"""
import os
from datetime import date

_today = date.today()
_default_end_year = _today.year if _today.month >= 11 else _today.year - 1

# GitHub Actionsから上書き可能（workflow_dispatchで年を指定する場合）
DATA_END_YEAR = int(os.environ.get("NPB_DATA_END_YEAR", _default_end_year))
TARGET_YEAR = DATA_END_YEAR + 1   # 予測対象年
DATA_START_YEAR = 2015


def displayed_years(exists):
    """(DATA_END_YEAR, TARGET_YEAR) for the app and the API.

    DATA_END_YEAR rolls over on 11/1, but the next season's predictions are
    published only by the 3/1 annual_update. Until they exist, keep showing the
    current ones: fall back one year when marcel_hitters_{TARGET_YEAR}.csv is known
    to be missing and the previous year's file is known to exist. ``exists(name)``
    returns True / False for a file in data/projections, or None when it could not
    check (then nothing changes). An explicit NPB_DATA_END_YEAR is always respected.
    """
    if "NPB_DATA_END_YEAR" in os.environ:
        return DATA_END_YEAR, TARGET_YEAR
    if (exists(f"marcel_hitters_{TARGET_YEAR}.csv") is False
            and exists(f"marcel_hitters_{TARGET_YEAR - 1}.csv") is True):
        return DATA_END_YEAR - 1, TARGET_YEAR - 1
    return DATA_END_YEAR, TARGET_YEAR
YEARS = list(range(DATA_START_YEAR, DATA_END_YEAR + 1))

# ベイズモデル関連パス
from pathlib import Path
_BASE_DIR = Path(__file__).parent
BAYES_DIR = _BASE_DIR / "data" / "bayes"
PROJECTIONS_DIR = _BASE_DIR / "data" / "projections"
