"""書き出しの門の検算（`_finalize_outputs` 単体 ＋ `main()` の呼び出し位置）。

`tests/test_sigma_model.py` は `resolve_sigma` 単体しか見ていない。この弧で
実際に壊れたのは**そこではなく門の置き場所**だった:

  - 検査と書き出しを投手側のブロックの中に入れてしまい、**投手が 0 行のとき
    fail-closed が丸ごと発動しなかった**（打者が平の σ に落ちても rc=0）
  - 打者 CSV を書く前に `Saved:` と印字していたので、**書けていないのに成功
    したと見えた**
  - 打者が 0 行・投手が非 0 行だと `UnboundLocalError` で投手 CSV も落ちた

🔴 **`_finalize_outputs` を直接呼ぶだけでは、この 3 つのどれも縛れない。**
壊れていたのは関数の中身ではなく `main()` の**呼び出し位置**だったので、
`main()` を実際に駆動する検査（`test_main_*`）が要る。最初に書いた 8 件は
関数を直接呼ぶだけで、門を投手ブロックへ戻す変異も、`main()` が
`_finalize_outputs` を呼ばなくなる変異も 1 つも捕まえなかった（監査で判明）。

🔴 **本物の予測は一切走らせない**（`data/projections/*_2026.csv` は 2026-03-23 に
凍結した採点の記録）。`OUT_DIR` を一時ディレクトリへ差し替え、**差し替わった
ことを assert してから** `main()` を呼ぶ。予測関数はすべて偽物に置き換える。

実行: python -m pytest tests/ -q   （pytest 無しなら python tests/test_output_gating.py）
"""
import contextlib
import io
import sys
import tempfile
import types
from pathlib import Path

import pandas as pd

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

import bayes_projection as B  # noqa: E402

HITTER_COLS = ["player", "team", "PA", "marcel_OPS", "stan_OPS", "bayes_OPS",
               "bayes_OPS_lo80", "bayes_OPS_hi80", "stan_delta", "method"]
PITCHER_COLS = ["player", "team", "IP", "marcel_ERA", "stan_ERA", "bayes_ERA",
                "bayes_ERA_lo80", "bayes_ERA_hi80", "stan_delta", "method"]


def _hitters(n=3, with_ci=True):
    return pd.DataFrame({
        "player": [f"打者{i}" for i in range(n)], "team": ["T"] * n,
        "PA": [300] * n, "marcel_OPS": [0.690] * n, "stan_OPS": [0.705] * n,
        "bayes_OPS": [0.700] * n,
        "bayes_OPS_lo80": [0.600] * n if with_ci else [None] * n,
        "bayes_OPS_hi80": [0.800] * n if with_ci else [None] * n,
        "stan_delta": [0.010] * n, "method": ["bma_jpn"] * n,
    })


def _pitchers(n=2, with_ci=True):
    return pd.DataFrame({
        "player": [f"投手{i}" for i in range(n)], "team": ["T"] * n,
        "IP": [80.0] * n, "marcel_ERA": [3.60] * n, "stan_ERA": [3.55] * n,
        "bayes_ERA": [3.50] * n,
        "bayes_ERA_lo80": [2.50] * n if with_ci else [None] * n,
        "bayes_ERA_hi80": [4.50] * n if with_ci else [None] * n,
        "stan_delta": [-0.05] * n, "method": ["bma_jpn"] * n,
    })


def _empty(cols=None):
    """実物に合わせた空枠。

    ⚠️ `predict_hitters` / `predict_pitchers` はデータが無いとき **列を 1 つも
    持たない** `pd.DataFrame()` を返す（bayes_projection.py:402, 514）。列つきの
    空枠で試すと、`_check_sigma_health` の `if len(df) == 0: continue` を消す変異が
    捕まらない（実物なら KeyError になる）。
    """
    return pd.DataFrame()


def _saved_lines(out):
    return [ln.split("Saved:", 1)[1].strip()
            for ln in out.splitlines() if ln.startswith("Saved:")]


# --------------------------------------------------------------------------
# A. _finalize_outputs 単体
# --------------------------------------------------------------------------

def _run(hitters, pitchers, write_hitters=True, write_pitchers=True, fallbacks=None):
    """一時ディレクトリへ書かせて (終了コード, 出来たファイル名, stdout) を返す。"""
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
            code, buf = 0, io.StringIO()
            try:
                with contextlib.redirect_stdout(buf):
                    B._finalize_outputs(hitters, hp, pitchers, pp)
            except SystemExit as exc:
                code = exc.code
            return code, sorted(p.name for p in Path(tmp).iterdir()), buf.getvalue()
    finally:
        B.SIGMA_FALLBACKS.clear()
        B.SIGMA_FALLBACKS.update(saved)


def test_nothing_is_written_when_the_gate_fails():
    code, files, _ = _run(_hitters(), _pitchers(), fallbacks={"broken_sigma_model": 1})
    assert code == 1, code
    assert files == [], files


def test_both_files_are_written_on_a_clean_run():
    code, files, _ = _run(_hitters(), _pitchers())
    assert code == 0, code
    assert files == ["bayes_hitters_TEST.csv", "bayes_pitchers_TEST.csv"], files


def test_saved_lines_match_the_files_that_exist():
    """🔴「Saved:」が嘘をつかないこと。

    以前ここは `assert "bayes_pitchers_TEST.csv" not in out` だったが、その
    シナリオでは投手の保存先が None なのでどの実装からも出得ず、**構成上ほぼ
    必ず真＝何も縛っていなかった**（監査で判明）。本数と実在で縛る。
    """
    with tempfile.TemporaryDirectory() as tmp:
        hp = Path(tmp) / "bayes_hitters_TEST.csv"
        buf = io.StringIO()
        with contextlib.redirect_stdout(buf):
            B._finalize_outputs(_hitters(), hp,
                                _empty(PITCHER_COLS), None)
        lines = _saved_lines(buf.getvalue())
        made = sorted(p.name for p in Path(tmp).iterdir())
        assert len(lines) == len(made), (lines, made)
        for path in lines:
            assert Path(path).exists(), path      # 「Saved: None」を捕まえる


def test_each_file_holds_the_frame_it_is_named_for():
    """枠と保存先を取り違えていないこと（ファイル名だけ見ていると素通りする）。"""
    with tempfile.TemporaryDirectory() as tmp:
        hp, pp = Path(tmp) / "h.csv", Path(tmp) / "p.csv"
        with contextlib.redirect_stdout(io.StringIO()):
            B._finalize_outputs(_hitters(), hp, _pitchers(), pp)
        head_h = hp.read_bytes()
        head_p = pp.read_bytes()
        assert b"bayes_OPS" in head_h and b"bayes_ERA" not in head_h, head_h[:120]
        assert b"bayes_ERA" in head_p and b"bayes_OPS" not in head_p, head_p[:120]
        # BOM つき utf-8（Excel 向け）で、無名の index 列を足していないこと
        assert head_h.startswith("﻿".encode("utf-8")), head_h[:8]
        assert head_h.decode("utf-8-sig").splitlines()[0].split(",")[0] == "player"


def test_saved_is_not_printed_when_the_write_fails():
    """書き出しが失敗したのに「Saved:」と印字しないこと。

    印字を `to_csv` の**前**へ動かす変異は、成功する走行では区別が付かない
    （どちらでもファイルは出来る）。失敗させて初めて落ちる。
    """
    with tempfile.TemporaryDirectory() as tmp:
        bad = Path(tmp) / "no_such_dir" / "h.csv"   # 親が無いので to_csv が失敗する
        buf, raised = io.StringIO(), None
        try:
            with contextlib.redirect_stdout(buf):
                B._finalize_outputs(_hitters(), bad, _pitchers(), Path(tmp) / "p.csv")
        except OSError as exc:
            raised = exc
        assert raised is not None, "書き出しの失敗が伝播していない"
        assert _saved_lines(buf.getvalue()) == [], buf.getvalue()


def test_rows_without_intervals_are_counted_from_the_shipped_frames():
    _, _, out = _run(_hitters(n=4, with_ci=False), _pitchers(n=2, with_ci=True))
    assert "区間あり 2" in out, out
    assert "区間なし 4" in out, out


# --------------------------------------------------------------------------
# B. main() の呼び出し位置（これが無いと BLOCKER の再発を検出できない）
# --------------------------------------------------------------------------

def _drive_main(hitters, pitchers, fallbacks=None):
    """main() を一時ディレクトリで駆動して (終了コード, ファイル名, stdout) を返す。

    🔴 本物の data/projections には絶対に書かせない。OUT_DIR を tmp へ差し替え、
    **差し替わったことを assert してから** main() を呼ぶ。
    """
    names = ("OUT_DIR", "PosteriorStore", "predict_hitters", "predict_pitchers",
             "predict_foreign_hitters", "predict_foreign_pitchers", "_filter_roster")
    orig = {k: getattr(B, k) for k in names}
    saved = dict(B.SIGMA_FALLBACKS)
    try:
        with tempfile.TemporaryDirectory() as tmp:
            B.OUT_DIR = Path(tmp)
            assert B.OUT_DIR != orig["OUT_DIR"], "OUT_DIR を差し替えられていない"
            assert str(B.OUT_DIR).startswith(tempfile.gettempdir()), B.OUT_DIR

            def fake_hitters(store):
                # main() は冒頭でカウンタを 0 に戻すので、積むのは呼ばれた後。
                for k, v in (fallbacks or {}).items():
                    B.SIGMA_FALLBACKS[k] = v
                B.SIGMA_FALLBACKS["used_sigma_model"] += len(hitters)
                return hitters

            def fake_pitchers(store):
                B.SIGMA_FALLBACKS["used_sigma_model"] += len(pitchers)
                return pitchers

            B.PosteriorStore = lambda *a, **k: types.SimpleNamespace(version="test")
            B.predict_hitters = fake_hitters
            B.predict_pitchers = fake_pitchers
            B.predict_foreign_hitters = lambda store: _empty(["player"])
            B.predict_foreign_pitchers = lambda store: _empty(["player"])
            B._filter_roster = lambda df: df

            code, buf = 0, io.StringIO()
            try:
                with contextlib.redirect_stdout(buf):
                    B.main()
            except SystemExit as exc:
                code = exc.code
            return code, sorted(p.name for p in Path(tmp).iterdir()), buf.getvalue()
    finally:
        for k, v in orig.items():
            setattr(B, k, v)
        B.SIGMA_FALLBACKS.clear()
        B.SIGMA_FALLBACKS.update(saved)


def test_main_gate_fires_when_pitchers_are_empty_and_hitters_fell_back():
    """🔴 これが監査 5 回目の BLOCKER そのもの。門を投手ブロックへ戻すと落ちる。"""
    code, files, _ = _drive_main(_hitters(), _empty(PITCHER_COLS),
                                 fallbacks={"bad_playing_time": 3})
    assert code == 1, code
    assert files == [], files


def test_main_writes_the_hitter_file_when_pitchers_are_empty():
    code, files, out = _drive_main(_hitters(), _empty(PITCHER_COLS))
    assert code == 0, code
    assert files == [f"bayes_hitters_{B.TARGET_YEAR}.csv"], files
    assert len(_saved_lines(out)) == len(files), out


def test_main_writes_the_pitcher_file_when_hitters_are_empty():
    # c58feb7 ではここが UnboundLocalError だった。
    code, files, out = _drive_main(_empty(HITTER_COLS), _pitchers())
    assert code == 0, code
    assert files == [f"bayes_pitchers_{B.TARGET_YEAR}.csv"], files
    assert len(_saved_lines(out)) == len(files), out


def test_main_writes_both_files_on_a_clean_run():
    code, files, out = _drive_main(_hitters(), _pitchers())
    assert code == 0, code
    assert files == [f"bayes_hitters_{B.TARGET_YEAR}.csv",
                     f"bayes_pitchers_{B.TARGET_YEAR}.csv"], files
    assert len(_saved_lines(out)) == 2, out


def test_main_fails_when_both_frames_are_empty():
    """上流が全滅したら黙って成功しない。

    以前は rc=0・ファイル 0 件・「Saved:」0 行で通っていた＝CI からは成功に
    見えるのに何も更新されない（前年の CSV がそのまま出荷される）。
    """
    code, files, _ = _drive_main(_empty(HITTER_COLS), _empty(PITCHER_COLS))
    assert code == 1, code
    assert files == [], files


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
