"""書き出しの門（_finalize_outputs）の検算。

`tests/test_sigma_model.py` は `resolve_sigma` 単体しか見ていない。この弧で
実際に壊れたのは**そこではなく門の置き場所**だった:

  - 検査と書き出しを投手側のブロックの中に入れてしまい、**投手が 0 行のとき
    fail-closed が丸ごと発動しなかった**（打者が平の σ に落ちても rc=0）
  - 打者 CSV を書く前に `Saved:` と印字していたので、**書けていないのに成功
    したと見えた**
  - 打者が 0 行・投手が非 0 行だと `UnboundLocalError` で投手 CSV も落ちた

このファイルはその 3 つを回帰として固定する。**本物の予測は一切走らせない**
（`data/projections/*_2026.csv` は 2026-03-23 に凍結した採点の記録）。枠は手で
作り、書き出し先は一時ディレクトリだけを渡す。

実行: python -m pytest tests/ -q   （pytest 無しなら python tests/test_output_gating.py）
"""
import sys
import tempfile
from pathlib import Path

import pandas as pd

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

import bayes_projection as B  # noqa: E402


def _hitters(n=3, with_ci=True):
    return pd.DataFrame({
        "player": [f"打者{i}" for i in range(n)],
        "bayes_OPS": [0.700] * n,
        "bayes_OPS_lo80": [0.600] * n if with_ci else [None] * n,
    })


def _pitchers(n=2, with_ci=True):
    return pd.DataFrame({
        "player": [f"投手{i}" for i in range(n)],
        "bayes_ERA": [3.50] * n,
        "bayes_ERA_lo80": [2.50] * n if with_ci else [None] * n,
    })


def _empty(cols):
    return pd.DataFrame({c: [] for c in cols})


def _run(hitters, pitchers, write_hitters=True, write_pitchers=True, fallbacks=None):
    """一時ディレクトリへ書かせて (SystemExit のコード, 実際に出来たファイル名) を返す。"""
    saved = dict(B.SIGMA_FALLBACKS)
    for k in B.SIGMA_FALLBACKS:
        B.SIGMA_FALLBACKS[k] = 0
    B.SIGMA_FALLBACKS["used_sigma_model"] = len(hitters) + len(pitchers)
    for k, v in (fallbacks or {}).items():
        B.SIGMA_FALLBACKS[k] = v
    try:
        with tempfile.TemporaryDirectory() as tmp:
            hp = Path(tmp) / "bayes_hitters_TEST.csv" if write_hitters else None
            pp = Path(tmp) / "bayes_pitchers_TEST.csv" if write_pitchers else None
            code = 0
            try:
                B._finalize_outputs(hitters, hp, pitchers, pp)
            except SystemExit as exc:
                code = exc.code
            return code, sorted(p.name for p in Path(tmp).iterdir())
    finally:
        B.SIGMA_FALLBACKS.clear()
        B.SIGMA_FALLBACKS.update(saved)


def test_gate_fires_even_when_pitchers_are_empty():
    # 🔴 これが c58feb7 の BLOCKER。投手が 0 行でも打者の fallback で止まること。
    code, files = _run(_hitters(), _empty(["player", "bayes_ERA", "bayes_ERA_lo80"]),
                       write_pitchers=False, fallbacks={"bad_playing_time": 3})
    assert code == 1, code
    assert files == [], files


def test_gate_fires_even_when_hitters_are_empty():
    code, files = _run(_empty(["player", "bayes_OPS", "bayes_OPS_lo80"]), _pitchers(),
                       write_hitters=False, fallbacks={"no_playing_time": 2})
    assert code == 1, code
    assert files == [], files


def test_nothing_is_written_when_the_gate_fails():
    # 両方そろっていても、fallback が 1 件でもあれば 1 ファイルも残さない。
    code, files = _run(_hitters(), _pitchers(), fallbacks={"broken_sigma_model": 1})
    assert code == 1, code
    assert files == [], files


def test_both_files_are_written_on_a_clean_run():
    code, files = _run(_hitters(), _pitchers())
    assert code == 0, code
    assert files == ["bayes_hitters_TEST.csv", "bayes_pitchers_TEST.csv"], files


def test_empty_pitchers_still_writes_the_hitter_file():
    # 片方が 0 行でも、もう片方は出荷される（c58feb7 では打者が消えていた）。
    code, files = _run(_hitters(), _empty(["player", "bayes_ERA", "bayes_ERA_lo80"]),
                       write_pitchers=False)
    assert code == 0, code
    assert files == ["bayes_hitters_TEST.csv"], files


def test_empty_hitters_still_writes_the_pitcher_file():
    # c58feb7 ではここが UnboundLocalError だった。
    code, files = _run(_empty(["player", "bayes_OPS", "bayes_OPS_lo80"]), _pitchers(),
                       write_hitters=False)
    assert code == 0, code
    assert files == ["bayes_pitchers_TEST.csv"], files


def test_saved_is_printed_only_for_files_that_exist(capsys=None):
    # 「Saved:」が嘘をつかないこと＝書いた枠の名前しか出ない。
    import io
    import contextlib
    buf = io.StringIO()
    with contextlib.redirect_stdout(buf):
        code, files = _run(_hitters(), _empty(["player", "bayes_ERA", "bayes_ERA_lo80"]),
                           write_pitchers=False)
    out = buf.getvalue()
    assert code == 0, code
    assert "bayes_hitters_TEST.csv" in out, out
    assert "bayes_pitchers_TEST.csv" not in out, out


def test_rows_without_intervals_are_counted_from_the_shipped_frames():
    # 区間なしの行数は出荷する枠から数える（カウンタの引き算で出さない）。
    import io
    import contextlib
    buf = io.StringIO()
    with contextlib.redirect_stdout(buf):
        _run(_hitters(n=4, with_ci=False), _pitchers(n=2, with_ci=True))
    out = buf.getvalue()
    assert "区間あり 2" in out, out
    assert "区間なし 4" in out, out


if __name__ == "__main__":
    failed = 0
    for name, fn in sorted(globals().items()):
        if name.startswith("test_") and callable(fn):
            try:
                fn()
                print("PASS %s" % name)
            except Exception as exc:          # 例外も失敗として数える
                failed += 1                   # （pytest と同じ扱いにする）
                print("FAIL %s: %s: %s" % (name, type(exc).__name__, exc))
    print("\n%d failed" % failed)
    sys.exit(1 if failed else 0)
