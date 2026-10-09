"""Animated 2026 pennant race against the March forecast: each team's wins
piling up by date, next to the 80% range of final wins that
team_sim_2026.csv (frozen 2026-03-23, before opening day) gave it.

Usage: python team_race_gif.py <games_2026.csv> <team_sim_2026.csv> <out gif>
"""
import datetime as dt
import io
import json
import pathlib
import sys

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
from PIL import Image

plt.rcParams["font.family"] = "Noto Sans CJK JP"
LEAGUES = {"CL": ["阪神", "巨人", "DeNA", "広島", "ヤクルト", "中日"],
           "PL": ["ソフトバンク", "西武", "日本ハム", "オリックス", "ロッテ", "楽天"]}
COLORS = ["#b08900", "#e8590c", "#1971c2", "#d6336c", "#2b8a3e", "#6741d9"]
LABEL_GAP = 4.2   # wins between stacked labels
STEP_DAYS = 3


def spread(values, gap):
    """Label heights: the values pushed apart to at least gap, kept centred."""
    order = np.argsort(values)
    pos = np.array(values, float)[order]
    for i in range(1, len(pos)):
        pos[i] = max(pos[i], pos[i - 1] + gap)
    pos -= (pos - np.array(values, float)[order]).mean()
    for i in range(1, len(pos)):
        pos[i] = max(pos[i], pos[i - 1] + gap)
    out = np.empty_like(pos)
    out[order] = pos
    return out


def main(games_csv, sim_csv, out_gif):
    games = pd.read_csv(games_csv)
    sim = pd.read_csv(sim_csv, encoding="utf-8-sig").set_index("team")
    games["date"] = pd.to_datetime(games["date"])
    teams = [t for lg in LEAGUES.values() for t in lg]
    if set(sim.index) != set(teams):
        raise SystemExit("predictions are not the 12 teams")
    # cumulative wins of every team at the end of every day
    days = pd.date_range(games["date"].min(), games["date"].max(), freq="D")
    wins = pd.DataFrame(0, index=days, columns=teams)
    for g in games.itertuples():
        if g.home_score != g.away_score:
            w = g.home if g.home_score > g.away_score else g.away
            wins.loc[g.date, w] += 1
    wins = wins.cumsum()
    played = pd.concat([games["home"], games["away"]]).value_counts()
    final = wins.iloc[-1]

    end = days[-1]
    shows = list(days[::STEP_DAYS])
    if shows[-1] != end:
        shows.append(end)
    durations = [80] * len(shows)
    durations[0] = 900
    durations[-1] = 6000
    inside = {t: bool(sim.loc[t, "wins_80ci_lo"] <= final[t] <= sim.loc[t, "wins_80ci_hi"]) for t in teams}
    k = sum(inside.values())
    ymax = max(final.max(), sim["wins_80ci_hi"].max()) + 6
    frames = []
    for day in shows:
        last = day == end
        fig, axes = plt.subplots(1, 2, figsize=(12, 6), dpi=80, sharey=True)
        fig.patch.set_facecolor("white")
        for ax, (lg, lteams) in zip(axes, LEAGUES.items()):
            ax.set_facecolor("white")
            order = sorted(lteams, key=lambda t: -sim.loc[t, "median_wins"])
            ys = spread([wins.loc[day, t] for t in lteams], LABEL_GAP)
            for i, (t, c) in enumerate(zip(lteams, COLORS)):
                w = wins.loc[:day, t]
                ax.plot(w.index, w.to_numpy(), color=c, lw=2.2)
                ax.text(day + pd.Timedelta(days=3), ys[i], f"{t} {int(w.iloc[-1])}", color=c, fontsize=10,
                        va="center", weight="bold")
                # the March forecast: 80% range of final wins, one bar per team side by side
                x = end + pd.Timedelta(days=68 + 5 * order.index(t))
                lo, hi, med = sim.loc[t, ["wins_80ci_lo", "wins_80ci_hi", "median_wins"]]
                ax.plot([x, x], [lo, hi], color=c, lw=5, alpha=0.35, solid_capstyle="butt")
                ax.plot([x - pd.Timedelta(days=2), x + pd.Timedelta(days=2)], [med, med], color=c, lw=2)
                if last:
                    ax.scatter([x], [final[t]], s=55, color=c, zorder=4,
                               marker="o" if inside[t] else "X", edgecolor="white", lw=0.8)
            ax.set_xlim(days[0] - pd.Timedelta(days=3), end + pd.Timedelta(days=100))
            ax.set_ylim(0, ymax)
            ax.set_title({"CL": "Central League", "PL": "Pacific League"}[lg], fontsize=14, loc="left")
            ticks = [dt.datetime(2026, m, 1) for m in range(4, 11)]
            ax.set_xticks(ticks)
            ax.set_xticklabels([f"{m}月" for m in range(4, 11)], fontsize=11)
            ax.tick_params(axis="y", labelsize=11)
            ax.text(end + pd.Timedelta(days=80), sim.loc[lteams, 'wins_80ci_lo'].min() - 7, "March\nforecast", ha="center", va="top",
                    fontsize=10, color="#555555")
            for side in ("top", "right"):
                ax.spines[side].set_visible(False)
        axes[0].set_ylabel("Wins", fontsize=14)
        head = (f"{k} of 12 teams finished inside the 80% range forecast in March" if last
                else "2026 wins, game by game, against the March forecast")
        fig.suptitle(head, fontsize=16, x=0.02, ha="left", y=0.975)
        fig.text(0.98, 0.93, day.strftime("%-m/%-d"), fontsize=24, ha="right", va="top", weight="bold",
                 color="#444444")
        fig.text(0.02, 0.012, "Bars: 80% range of final wins from team_sim_2026.csv (frozen 2026-03-23); "
                 "tick = median. End: o inside, x outside. Results: npb.jp.", fontsize=9.5, color="#555555")
        fig.subplots_adjust(left=0.07, right=0.98, top=0.8, bottom=0.1, wspace=0.12)
        buf = io.BytesIO()
        fig.savefig(buf, format="png", facecolor="white")
        plt.close(fig)
        frames.append(Image.open(buf).convert("RGB"))

    pal = frames[-1].quantize(colors=64, method=Image.Quantize.MEDIANCUT)  # last frame holds every colour
    q = [fr.quantize(palette=pal, dither=Image.Dither.NONE) for fr in frames]
    q[0].save(out_gif, save_all=True, append_images=q[1:], duration=durations, loop=0, optimize=True)
    print(json.dumps({"frames": len(q), "seconds": sum(durations) / 1000, "bytes": pathlib.Path(out_gif).stat().st_size,
                      "last_day": end.date().isoformat(), "games": {t: int(played[t]) for t in teams},
                      "k_inside_80": k}, ensure_ascii=False))


if __name__ == "__main__":
    if len(sys.argv) != 4:
        raise SystemExit(__doc__)
    main(*sys.argv[1:])
