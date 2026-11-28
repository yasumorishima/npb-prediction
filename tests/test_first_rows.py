"""ml_projection._first_rows returns, for every key, the same row as the
full-table mask it replaced: frame[(frame[k1] == a) & (frame[k2] == b)].iloc[0]."""
import sys
from pathlib import Path

import numpy as np
import pandas as pd

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
import ml_projection as M  # noqa: E402


def _frame():
    rng = np.random.default_rng(3)
    n = 400
    df = pd.DataFrame({
        "player": rng.choice(["山田　太郎", "佐藤", "鈴木", "ロペス", None], n),
        "year": rng.choice([2021, 2022, 2023, 2024], n),
        "team": rng.choice(["阪神", "巨人", "広島"], n),
        "PA": rng.integers(0, 600, n),
        "OPS": rng.random(n),
    })
    df.loc[5, "OPS"] = np.nan
    return df                                  # many duplicate (player, year): traded players


def test_same_row_as_the_mask():
    df = _frame()
    first = M._first_rows(df, ["player", "year"])
    keys = [(p, y) for p in ["山田　太郎", "佐藤", "鈴木", "ロペス", "いない", None, np.nan] for y in [2020, 2021, 2022, 2023, 2024]]
    for p, y in keys:
        pdata = df[(df["player"] == p) & (df["year"] == y)]
        got = first.get((p, y))
        if len(pdata) == 0:
            assert got is None, (p, y)
        else:
            pd.testing.assert_series_equal(got, pdata.iloc[0], check_names=True)
    assert sum(len(df[(df["player"] == p) & (df["year"] == y)]) > 1 for p, y in keys) > 5   # duplicates exercised


def test_numpy_year_keys_match():
    df = _frame()
    first = M._first_rows(df, ["player", "year"])
    yr = df["year"].unique()[0]                 # numpy int, as in the builders (target_year - 1)
    assert first.get(("佐藤", yr - 1)) is not None or len(df[(df["player"] == "佐藤") & (df["year"] == yr - 1)]) == 0
    assert first.get(("佐藤", np.int64(2022))) is first.get(("佐藤", 2022))
