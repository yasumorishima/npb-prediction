"""Game-by-game results of the 2026 NPB regular season from npb.jp, checked
against the baseball-data.com standings table.

Usage: python games26.py <out dir>
Writes games_2026.csv (date, home, away, home_score, away_score) and
standings_2026.csv (the standings table as fetched, with the fetch time).
Refuses to write if any team's W / L / T from the games differs from the
standings table.
"""
import datetime as dt
import io
import re
import sys
import time
from pathlib import Path

import pandas as pd
import requests
from bs4 import BeautifulSoup

HEADERS = {"User-Agent": "Mozilla/5.0 (compatible; research-bot/1.0)"}
TEAMS = ["阪神", "DeNA", "巨人", "中日", "広島", "ヤクルト",
         "ソフトバンク", "日本ハム", "オリックス", "楽天", "西武", "ロッテ"]
SEASON_GAMES = 143
GAME = re.compile(r"^(\S+) (\d+) - (\d+) (\S+)$")
DATE = re.compile(r"^(\d{1,2})/(\d{1,2})")


def get(url):
    r = requests.get(url, headers=HEADERS, timeout=30)
    r.raise_for_status()
    return r.content


def main(out_dir):
    out = Path(out_dir)
    out.mkdir(parents=True, exist_ok=True)
    rows, unknown = [], set()
    for month in range(3, 11):
        url = f"https://npb.jp/games/2026/schedule_{month:02d}_detail.html"
        r = requests.get(url, headers=HEADERS, timeout=30)
        if r.status_code == 404:
            continue
        r.raise_for_status()
        r.encoding = "utf-8"
        day = None
        for tr in BeautifulSoup(r.text, "lxml").select("tr"):
            cells = [c.get_text(" ", strip=True) for c in tr.find_all(["td", "th"])]
            if not cells:
                continue
            m = DATE.match(cells[0])
            if m:
                day = dt.date(2026, int(m.group(1)), int(m.group(2)))
                cells = cells[1:]
            if not cells or day is None:
                continue
            g = GAME.match(cells[0])
            if not g:
                continue
            home, hs, as_, away = g.group(1), int(g.group(2)), int(g.group(3)), g.group(4)
            if {home, away} == {"セ・リーグ", "パ・リーグ"}:   # all-star games
                continue
            if home not in TEAMS or away not in TEAMS:
                unknown.add(cells[0])
                continue
            rows.append({"date": day.isoformat(), "home": home, "away": away, "home_score": hs, "away_score": as_})
        time.sleep(1.5)
    if unknown:
        raise SystemExit(f"unrecognised game rows: {sorted(unknown)[:5]}")
    games = pd.DataFrame(rows).sort_values("date", kind="stable").reset_index(drop=True)
    # regular season only: the October page also lists the Climax Series, whose
    # teams have already played their 143. A game counts while both teams are short of 143.
    played = {t: 0 for t in TEAMS}
    keep = []
    for g in games.itertuples():
        ok = played[g.home] < SEASON_GAMES and played[g.away] < SEASON_GAMES
        keep.append(ok)
        if ok:
            played[g.home] += 1
            played[g.away] += 1
    dropped = games[[not k for k in keep]]
    games = games[keep].reset_index(drop=True)
    if len(dropped) and dropped["date"].min() <= games["date"].max():
        raise SystemExit("a game past 143 falls before the last regular-season game")
    games.to_csv(out / "games_2026_unchecked.csv", index=False)

    fetched = dt.datetime.now(dt.timezone.utc).isoformat(timespec="seconds")
    st = pd.concat(pd.read_html(io.BytesIO(get("https://baseball-data.com/team/standings.html")), flavor="lxml", encoding="utf-8")[:2])
    st.columns = [str(c).replace(" ", "") for c in st.columns]
    st = st.rename(columns={"順位": "rank", "チーム": "team", "試合": "G", "勝利": "W", "敗北": "L", "引分": "T",
                            "残試合": "left"})
    st = st[["rank", "team", "G", "W", "L", "T", "left"]].assign(fetched_utc=fetched)

    rec = {t: [0, 0, 0] for t in TEAMS}
    for g in games.itertuples():
        if g.home_score > g.away_score:
            rec[g.home][0] += 1; rec[g.away][1] += 1
        elif g.home_score < g.away_score:
            rec[g.home][1] += 1; rec[g.away][0] += 1
        else:
            rec[g.home][2] += 1; rec[g.away][2] += 1
    bad = [(r.team, rec[r.team], (r.W, r.L, r.T)) for r in st.itertuples() if tuple(rec[r.team]) != (r.W, r.L, r.T)]
    if len(st) != 12 or set(st["team"]) != set(TEAMS) or bad:
        raise SystemExit(f"games do not reproduce the standings: {bad}")
    games.to_csv(out / "games_2026.csv", index=False)
    st.to_csv(out / "standings_2026.csv", index=False)
    print(f"{len(games)} games, last {games['date'].max()}, remaining {int(st['left'].sum())}, fetched {fetched}")


if __name__ == "__main__":
    if len(sys.argv) != 2:
        raise SystemExit(__doc__)
    main(sys.argv[1])
