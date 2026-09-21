"""resolve_sigma と区間の定数の検算。

このリポジトリで最初のテスト。狙いは 2 つだけ:
  1. Z80/Z95 が本当に 80%/95% の分位点か（外部の値でなく Φ を計算して確かめる）
  2. resolve_sigma が壊れた入力で pipeline を止めず、幅ゼロの区間も出さないこと

実行: python -m pytest tests/ -q   （pytest 無しなら python tests/test_sigma_model.py）
"""
import copy
import json
import math
import sys
from pathlib import Path

import numpy as np

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from bayes_projection import Z80, Z95, resolve_sigma, woba_to_ops_approx  # noqa: E402

POSTERIORS = json.loads(
    (Path(__file__).resolve().parents[1] / "data" / "bayes" / "posteriors.json")
    .read_text(encoding="utf-8")
)
HIT = POSTERIORS["jpn_hitter"]
PIT = POSTERIORS["jpn_pitcher"]


def _phi(z):
    """標準正規の累積分布（scipy に依存しない）。"""
    return 0.5 * (1.0 + math.erf(z / math.sqrt(2.0)))


def test_z_constants_are_the_quantiles_they_claim():
    # 片側ずつ 10% / 2.5% を残す
    assert abs(_phi(Z80) - 0.90) < 1e-10, _phi(Z80)
    assert abs(_phi(Z95) - 0.975) < 1e-10, _phi(Z95)


def test_woba_to_ops_is_affine_so_shifting_commutes_with_converting():
    # 区間を wOBA 空間で作って OPS へ変換しても、OPS 空間で同じ量だけ動かしても
    # 同じ結果になること（打者の再センタリングがこの性質に乗っている）
    a, b, shift_woba = 0.300, 0.360, 0.012
    lhs = woba_to_ops_approx(a + shift_woba) - woba_to_ops_approx(b + shift_woba)
    rhs = woba_to_ops_approx(a) - woba_to_ops_approx(b)
    assert abs(lhs - rhs) < 1e-12


def test_falls_back_to_flat_sigma_for_unusable_playing_time():
    for bad in (None, 0, -5, float("nan"), float("inf"), "abc", [], {}):
        assert resolve_sigma(HIT, bad) == HIT["sigma_residual"], bad
        assert resolve_sigma(PIT, bad) == PIT["sigma_residual"], bad


def test_falls_back_when_sigma_model_is_broken_rather_than_raising():
    for key in ("sigma_base", "gamma", "log_mean", "log_sd"):
        broken = copy.deepcopy(HIT)
        del broken["sigma_model"][key]
        try:
            got = resolve_sigma(broken, 300)
        except Exception as exc:
            raise AssertionError(
                "resolve_sigma raised %s when sigma_model lost %r; it must fall back"
                % (type(exc).__name__, key))
        assert got == HIT["sigma_residual"], (key, got)


def test_never_returns_a_zero_width_sigma():
    for key, value in (("log_sd", 0.0), ("log_sd", -1.0), ("sigma_base", 0.0)):
        broken = copy.deepcopy(HIT)
        broken["sigma_model"][key] = value
        got = resolve_sigma(broken, 300)
        assert got > 0, (key, value, got)
        assert got == HIT["sigma_residual"], (key, value, got)


def test_sigma_shrinks_as_playing_time_grows():
    # gamma < 0 であることを、係数でなく振る舞いで確かめる
    assert resolve_sigma(HIT, 600) < resolve_sigma(HIT, 200) < resolve_sigma(HIT, 100)
    assert resolve_sigma(PIT, 180) < resolve_sigma(PIT, 60) < resolve_sigma(PIT, 30)


def test_never_extrapolates_below_the_fit_support():
    # fit は打者 PA>=100 / 投手 IP>=30 でしか行っていない。その下では
    # 下端の値で頭打ちになり、exp で外へ伸びないこと
    # キーを読まずに振る舞いで見る（pt_floor を消す変異でも落ちるように）
    at_floor_h = resolve_sigma(HIT, 100)
    at_floor_p = resolve_sigma(PIT, 30)
    for pt in (1, 5, 10, 30, 99):
        assert resolve_sigma(HIT, pt) == at_floor_h, pt
    for pt in (1, 5, 10, 29):
        assert resolve_sigma(PIT, pt) == at_floor_p, pt
    # 頭打ちにしたおかげで、区間が旧来の平の σ より広くなることはない
    assert at_floor_h < HIT["sigma_residual"]
    assert at_floor_p < PIT["sigma_residual"]


def test_sigma_model_matches_the_documented_ops_scale():
    # 打者の sigma_base は wOBA 尺度。2.33 倍して OPS 尺度の fit 値に戻ること
    ops_base = HIT["sigma_model"]["sigma_base"] * 2.33
    assert abs(ops_base - 0.093913) < 5e-6, ops_base


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
