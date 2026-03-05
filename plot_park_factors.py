"""
NPB パークファクター可視化スクリプト

12球団の単年PF（棒グラフ）と5年平均PF（折れ線）を年度別に表示する。
改修タイミングは縦線で明示（過去改修=オレンジ実線、予定=紫点線）。
楽天・バンテリン等、複数改修がある球場はすべての改修線を表示する。

出力: images/npb_park_factors_trend.png
"""

import warnings
warnings.filterwarnings("ignore")

from pathlib import Path
import pandas as pd
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import matplotlib.patches as mpatches
import matplotlib.lines as mlines

try:
    import japanize_matplotlib  # noqa: F401
except ImportError:
    pass  # フォント未インストール時はシステムデフォルトを使用

# ==================== 定数 ====================

DATA_START = 2016
DATA_END = 2025
XLIM = (DATA_START - 0.6, DATA_END + 0.7)
YLIM = (0.68, 1.52)

COLOR_HIGH   = "#3B82F6"   # 青: PF >= 1.0 （打者有利）
COLOR_LOW    = "#EF4444"   # 赤: PF < 1.0  （投手有利）
COLOR_LINE   = "#1F2937"   # 黒: 5年平均PF 折れ線
COLOR_REF    = "#9CA3AF"   # グレー: PF=1.0 基準線
COLOR_RENO   = "#F97316"   # オレンジ: 過去改修縦線
COLOR_FUTURE = "#9333EA"   # 紫: 改修予定縦線

# calc_park_factors.py の RENOVATION_BREAKS と同じ内容
RENOVATION_BREAKS: dict[str, list[int]] = {
    "ソフトバンク": [2015],        # HRテラス設置（左右中間▲6m）
    "ロッテ":       [2019],        # HRラグーン設置（左右中間▲4m）
    "日本ハム":     [2023],        # エスコンフィールド移転
    "楽天":         [2016, 2026],  # 2016: 天然芝化 / 2026: フェンス前方移設
    "中日":         [2026],        # HRウイング設置（左右中間▲6m）
}

# 改修ラベル（短め）
RENO_LABELS: dict[int, str] = {
    2015: "HRテラス",
    2016: "天然芝化",
    2019: "HRラグーン",
    2023: "エスコン開場",
    2026: "改修予定",
}

TEAMS = [
    "阪神", "DeNA", "巨人",
    "中日", "広島", "ヤクルト",
    "ソフトバンク", "日本ハム", "オリックス",
    "楽天", "西武", "ロッテ",
]

# ==================== データ読み込み ====================

def load_data() -> pd.DataFrame:
    proj_dir = Path(__file__).parent / "data" / "projections"
    path = proj_dir / "npb_park_factors.csv"
    if not path.exists():
        raise FileNotFoundError(
            f"{path} が見つかりません。先に calc_park_factors.py を実行してください。"
        )
    df = pd.read_csv(path, encoding="utf-8-sig")
    df["year"] = df["year"].astype(int)
    return df

# ==================== 改修縦線を描画 ====================

def _draw_renovation_lines(ax, team: str) -> None:
    """改修タイミングを縦線とラベルで表示する。
    - データ範囲外・過去（例: 2015）: 最初の棒の左端に「※YYYY年改修済」注釈
    - データ範囲内（例: 2019, 2023）: 改修前後をまたぐ橙色縦線
    - データ範囲後の予定（例: 2026）: 右端に紫色点線
    """
    breaks = RENOVATION_BREAKS.get(team, [])
    for reno_year in breaks:
        label = RENO_LABELS.get(reno_year, f"{reno_year}年改修")

        if reno_year < DATA_START:
            # データ範囲より前の改修 → 最初の棒の右上に小さく注釈
            ax.text(
                DATA_START + 0.0, YLIM[1] - 0.03,
                f"※{reno_year}年\n{label}済",
                fontsize=7, color=COLOR_RENO, ha="center", va="top",
                bbox=dict(boxstyle="round,pad=0.2", facecolor="white",
                          edgecolor=COLOR_RENO, alpha=0.85, linewidth=0.8),
                zorder=5,
            )

        elif reno_year > DATA_END:
            # データ範囲より後の改修予定 → 右端に点線
            x_pos = DATA_END + 0.45
            ax.axvline(x=x_pos, color=COLOR_FUTURE, linestyle=":",
                       linewidth=2.2, alpha=0.9, zorder=3)
            ax.text(
                x_pos - 0.08, YLIM[0] + 0.05,
                f"{reno_year}年\n{label}",
                fontsize=7, color=COLOR_FUTURE, ha="right", va="bottom",
                bbox=dict(boxstyle="round,pad=0.2", facecolor="white",
                          edgecolor=COLOR_FUTURE, alpha=0.85, linewidth=0.8),
                zorder=5,
            )

        else:
            # データ範囲内の改修 → 改修年の直前に実線
            x_line = reno_year - 0.5
            ax.axvline(x=x_line, color=COLOR_RENO, linestyle="-",
                       linewidth=2.2, alpha=0.9, zorder=3)
            ax.text(
                x_line + 0.1, YLIM[0] + 0.05,
                f"{reno_year}年\n{label}",
                fontsize=7, color=COLOR_RENO, ha="left", va="bottom",
                bbox=dict(boxstyle="round,pad=0.2", facecolor="white",
                          edgecolor=COLOR_RENO, alpha=0.85, linewidth=0.8),
                zorder=5,
            )

# ==================== メイン ====================

def main() -> None:
    df = load_data()
    # 最新年の球場名（タイトル用）
    stadium_map = df.groupby("team")["stadium"].last().to_dict()

    fig, axes = plt.subplots(4, 3, figsize=(20, 18))
    fig.patch.set_facecolor("#FAFAFA")
    fig.suptitle(
        "NPB 球場別 パークファクター年度推移（2016〜2025）\n"
        "棒グラフ: 単年PF ／ 折れ線: 5年平均PF ／ 縦線: 改修タイミング",
        fontsize=18, fontweight="bold", y=1.01,
    )

    all_years = list(range(DATA_START, DATA_END + 1))

    for idx, team in enumerate(TEAMS):
        ax = axes[idx // 3][idx % 3]
        ax.set_facecolor("#F8F9FA")

        team_df = df[df["team"] == team].sort_values("year")
        if team_df.empty:
            ax.set_visible(False)
            continue

        years  = team_df["year"].values
        pf     = team_df["PF"].values
        pf_5yr = team_df["PF_5yr"].values

        # ---- 棒グラフ（単年PF）----
        bar_colors = [COLOR_HIGH if p >= 1.0 else COLOR_LOW for p in pf]
        ax.bar(years, pf, color=bar_colors, alpha=0.65, width=0.65, zorder=2)

        # ---- 折れ線（5年平均PF）----
        valid_mask = ~pd.isna(pf_5yr)
        if valid_mask.sum() > 1:
            ax.plot(
                years[valid_mask], pf_5yr[valid_mask],
                color=COLOR_LINE, linewidth=2, marker="o", markersize=4,
                zorder=4,
            )

        # ---- 基準線 PF=1.0 ----
        ax.axhline(y=1.0, color=COLOR_REF, linestyle="--",
                   linewidth=1.2, zorder=1)

        # ---- 改修縦線 ----
        _draw_renovation_lines(ax, team)

        # ---- 軸設定 ----
        stadium = stadium_map.get(team, "")
        ax.set_title(f"{team}（{stadium}）", fontsize=14, fontweight="bold", pad=5)
        ax.set_xlim(XLIM)
        ax.set_ylim(YLIM)
        ax.set_xticks(all_years)
        ax.set_xticklabels(
            [f"'{str(y)[2:]}" for y in all_years],
            fontsize=9,
        )
        ax.set_ylabel("パークファクター", fontsize=11)
        ax.tick_params(axis="y", labelsize=10)
        ax.grid(axis="y", alpha=0.3, linestyle=":")
        ax.spines["top"].set_visible(False)
        ax.spines["right"].set_visible(False)

        # PF値ラベル（5年平均のみ・省スペース）
        for y, p5 in zip(years[valid_mask], pf_5yr[valid_mask]):
            ax.text(
                y, p5 + 0.025, f"{p5:.3f}",
                ha="center", va="bottom", fontsize=7, color=COLOR_LINE, zorder=6,
            )

    # ==================== 凡例（図全体の下部）====================
    handles = [
        mpatches.Patch(color=COLOR_HIGH, alpha=0.65, label="単年PF（打者有利: ≥1.00）"),
        mpatches.Patch(color=COLOR_LOW,  alpha=0.65, label="単年PF（投手有利: <1.00）"),
        mlines.Line2D([0], [0], color=COLOR_LINE, linewidth=2, marker="o",
                      markersize=5, label="5年平均PF"),
        mlines.Line2D([0], [0], color=COLOR_REF, linewidth=1.2, linestyle="--",
                      label="PF = 1.00（基準）"),
        mlines.Line2D([0], [0], color=COLOR_RENO, linewidth=2.2,
                      label="改修タイミング（実施済）"),
        mlines.Line2D([0], [0], color=COLOR_FUTURE, linewidth=2.2, linestyle=":",
                      label="改修予定（2026年）"),
    ]
    fig.legend(
        handles=handles,
        loc="lower center",
        ncol=3,
        fontsize=11,
        bbox_to_anchor=(0.5, -0.04),
        frameon=True,
        framealpha=0.9,
    )

    plt.tight_layout()

    out_dir = Path(__file__).parent / "images"
    out_dir.mkdir(exist_ok=True)
    out_path = out_dir / "npb_park_factors_trend.png"
    plt.savefig(out_path, dpi=150, bbox_inches="tight", facecolor="#FAFAFA")
    plt.close(fig)
    print(f"保存: {out_path}")


if __name__ == "__main__":
    main()
