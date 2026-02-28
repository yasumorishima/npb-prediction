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
YEARS = list(range(DATA_START_YEAR, DATA_END_YEAR + 1))
