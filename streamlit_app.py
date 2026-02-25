"""
NPB成績予測 Streamlitダッシュボード

Marcel法・LightGBM/XGBoost・ピタゴラス勝率・wOBA/wRC+の予測結果をブラウザで閲覧。

Data sources:
- プロ野球データFreak (https://baseball-data.com)
- 日本野球機構 NPB (https://npb.jp)
"""

import pandas as pd
import plotly.express as px
import plotly.graph_objects as go
import streamlit as st

BASE_URL = "https://raw.githubusercontent.com/yasumorishima/npb-prediction/master/"

NPB_TEAM_COLORS = {
    "DeNA": "#0055A5",
    "巨人": "#F97709",
    "阪神": "#FFE201",
    "広島": "#EE1C25",
    "中日": "#00468B",
    "ヤクルト": "#006AB6",
    "ソフトバンク": "#F5C70E",
    "日本ハム": "#004B97",
    "楽天": "#860029",
    "ロッテ": "#000000",
    "オリックス": "#C4A400",
    "西武": "#102A6F",
}

TEAMS = list(NPB_TEAM_COLORS.keys())


def _norm(name: str) -> str:
    return name.replace("\u3000", " ").strip()


@st.cache_data(ttl=3600)
def load_csv(path: str) -> pd.DataFrame:
    url = BASE_URL + path
    try:
        df = pd.read_csv(url, encoding="utf-8-sig")
    except Exception:
        return pd.DataFrame()
    if "player" in df.columns:
        df["player"] = df["player"].apply(_norm)
    if "team" in df.columns:
        df["team"] = df["team"].apply(_norm)
    return df


def load_all():
    return {
        "marcel_hitters": load_csv("data/projections/marcel_hitters_2026.csv"),
        "marcel_pitchers": load_csv("data/projections/marcel_pitchers_2026.csv"),
        "ml_hitters": load_csv("data/projections/ml_hitters_2026.csv"),
        "ml_pitchers": load_csv("data/projections/ml_pitchers_2026.csv"),
        "sabermetrics": load_csv("data/projections/npb_sabermetrics_2015_2025.csv"),
        "pythagorean": load_csv("data/projections/pythagorean_2015_2025.csv"),
    }


def _search(df: pd.DataFrame, name: str) -> pd.DataFrame:
    q = _norm(name)
    return df[df["player"].str.contains(q, na=False)]


def _pythagorean_wpct(rs: float, ra: float, k: float = 1.72) -> float:
    if ra == 0:
        return 1.0
    return rs**k / (rs**k + ra**k)


# --- ページ実装 ---


def page_hitter_prediction(data: dict):
    st.header("打者予測（2026年）")
    name = st.text_input("選手名で検索（部分一致）", key="hitter_search", placeholder="例: 牧、近藤、岡本")
    if not name:
        st.info("選手名を入力してください")
        return

    marcel = _search(data["marcel_hitters"], name)
    ml = _search(data["ml_hitters"], name)
    if marcel.empty and ml.empty:
        st.warning(f"「{name}」に該当する選手が見つかりません")
        return

    for _, row in marcel.iterrows():
        st.subheader(f"{row['player']}（{row['team']}）")
        cols = st.columns(6)
        cols[0].metric("OPS", f"{row['OPS']:.3f}")
        cols[1].metric("打率", f"{row['AVG']:.3f}")
        cols[2].metric("出塁率", f"{row['OBP']:.3f}")
        cols[3].metric("長打率", f"{row['SLG']:.3f}")
        cols[4].metric("本塁打", f"{row['HR']:.1f}")
        cols[5].metric("打点", f"{row['RBI']:.1f}")

        ml_match = ml[ml["player"] == row["player"]]
        if not ml_match.empty:
            ml_ops = ml_match.iloc[0]["pred_OPS"]
            st.markdown("**Marcel vs ML 比較**")
            fig = go.Figure(data=[
                go.Bar(name="Marcel", x=["OPS"], y=[row["OPS"]], marker_color="#4CAF50"),
                go.Bar(name="ML", x=["OPS"], y=[ml_ops], marker_color="#2196F3"),
            ])
            fig.update_layout(barmode="group", template="plotly_white", height=300,
                              yaxis_title="OPS", yaxis_range=[0, max(row["OPS"], ml_ops) * 1.2])
            st.plotly_chart(fig, use_container_width=True)
        st.divider()


def page_pitcher_prediction(data: dict):
    st.header("投手予測（2026年）")
    name = st.text_input("選手名で検索（部分一致）", key="pitcher_search", placeholder="例: 今永、山本、佐々木")
    if not name:
        st.info("選手名を入力してください")
        return

    marcel = _search(data["marcel_pitchers"], name)
    ml = _search(data["ml_pitchers"], name)
    if marcel.empty and ml.empty:
        st.warning(f"「{name}」に該当する選手が見つかりません")
        return

    for _, row in marcel.iterrows():
        st.subheader(f"{row['player']}（{row['team']}）")
        cols = st.columns(6)
        cols[0].metric("防御率", f"{row['ERA']:.2f}")
        cols[1].metric("WHIP", f"{row['WHIP']:.2f}")
        cols[2].metric("勝利", f"{row['W']:.1f}")
        cols[3].metric("敗北", f"{row['L']:.1f}")
        cols[4].metric("奪三振", f"{row['SO']:.1f}")
        cols[5].metric("投球回", f"{row['IP']:.1f}")

        ml_match = ml[ml["player"] == row["player"]]
        if not ml_match.empty:
            ml_era = ml_match.iloc[0]["pred_ERA"]
            st.markdown("**Marcel vs ML 比較**")
            fig = go.Figure(data=[
                go.Bar(name="Marcel", x=["ERA"], y=[row["ERA"]], marker_color="#4CAF50"),
                go.Bar(name="ML", x=["ERA"], y=[ml_era], marker_color="#2196F3"),
            ])
            fig.update_layout(barmode="group", template="plotly_white", height=300,
                              yaxis_title="ERA", yaxis_range=[0, max(row["ERA"], ml_era) * 1.3])
            st.plotly_chart(fig, use_container_width=True)
        st.divider()


def page_team_wpct(data: dict):
    st.header("チーム勝率（ピタゴラス勝率）")
    pyth = data["pythagorean"]
    if pyth.empty:
        st.error("データが読み込めませんでした")
        return

    col1, col2 = st.columns(2)
    team = col1.selectbox("チーム", TEAMS, key="team_wpct")
    year = col2.slider("年度", 2015, 2025, 2025, key="team_year")

    mask = pyth["team"].str.contains(_norm(team), na=False) & (pyth["year"] == year)
    matched = pyth[mask]
    if matched.empty:
        st.warning(f"{team} ({year}) のデータがありません")
        return

    row = matched.iloc[0]
    cols = st.columns(4)
    cols[0].metric("実際の勝率", f"{row['actual_WPCT']:.3f}")
    cols[1].metric("ピタゴラス勝率", f"{row['pyth_WPCT_npb']:.3f}")
    cols[2].metric("実際の勝数", f"{int(row['W'])}勝{int(row['L'])}敗")
    cols[3].metric("期待勝数", f"{row['pyth_W_npb']:.1f}", delta=f"{row['diff_W_npb']:+.1f}")

    st.markdown("**得失点**")
    fig = go.Figure(data=[
        go.Bar(name="得点", x=["得失点"], y=[row["RS"]], marker_color="#4CAF50"),
        go.Bar(name="失点", x=["得失点"], y=[row["RA"]], marker_color="#F44336"),
    ])
    fig.update_layout(barmode="group", template="plotly_white", height=300)
    st.plotly_chart(fig, use_container_width=True)

    fig2 = go.Figure(data=[
        go.Bar(name="実際の勝数", x=["勝数"], y=[row["W"]], marker_color=NPB_TEAM_COLORS.get(team, "#333")),
        go.Bar(name="期待勝数", x=["勝数"], y=[row["pyth_W_npb"]], marker_color="#9E9E9E"),
    ])
    fig2.update_layout(barmode="group", template="plotly_white", height=300, yaxis_title="勝数")
    st.plotly_chart(fig2, use_container_width=True)


def page_sabermetrics(data: dict):
    st.header("セイバーメトリクス（wOBA / wRC+ / wRAA）")
    saber = data["sabermetrics"]
    if saber.empty:
        st.error("データが読み込めませんでした")
        return

    col1, col2 = st.columns([2, 1])
    name = col1.text_input("選手名で検索", key="saber_search", placeholder="例: 近藤、牧")
    years = sorted(saber["year"].unique())
    year_option = col2.selectbox("年度", ["全年度"] + [str(int(y)) for y in years], key="saber_year")

    if not name:
        st.info("選手名を入力してください")
        return

    matched = _search(saber, name)
    if year_option != "全年度":
        matched = matched[matched["year"] == int(year_option)]

    if matched.empty:
        st.warning(f"「{name}」に該当するデータがありません")
        return

    display_cols = ["player", "team", "year", "PA", "wOBA", "wRC+", "wRAA", "AVG", "OBP", "SLG"]
    available = [c for c in display_cols if c in matched.columns]
    st.dataframe(
        matched[available].sort_values("year", ascending=False).reset_index(drop=True),
        use_container_width=True,
        hide_index=True,
    )

    if len(matched) > 1:
        player_name = matched.iloc[0]["player"]
        player_data = matched[matched["player"] == player_name].sort_values("year")
        if len(player_data) > 1:
            st.markdown(f"**{player_name} wRC+ 推移**")
            fig = px.line(player_data, x="year", y="wRC+", markers=True, template="plotly_white")
            fig.add_hline(y=100, line_dash="dash", line_color="gray",
                          annotation_text="リーグ平均 (100)")
            fig.update_layout(height=350, xaxis_title="年度", yaxis_title="wRC+")
            st.plotly_chart(fig, use_container_width=True)


def page_hitter_rankings(data: dict):
    st.header("打者ランキング（Marcel法 2026予測）")
    mh = data["marcel_hitters"]
    if mh.empty:
        st.error("データが読み込めませんでした")
        return

    col1, col2 = st.columns(2)
    top_n = col1.slider("表示人数", 5, 50, 20, key="hitter_rank_n")
    sort_by = col2.selectbox("ソート", ["OPS", "AVG", "HR", "RBI"], key="hitter_rank_sort")

    df = mh[mh["PA"] >= 200].sort_values(sort_by, ascending=False).head(top_n).reset_index(drop=True)
    df.index = df.index + 1
    df.index.name = "順位"

    display_cols = ["player", "team", "OPS", "AVG", "HR", "RBI", "PA"]
    available = [c for c in display_cols if c in df.columns]
    st.dataframe(df[available], use_container_width=True)

    fig = px.bar(
        df.head(top_n),
        y="player",
        x=sort_by,
        color="team",
        color_discrete_map=NPB_TEAM_COLORS,
        orientation="h",
        template="plotly_white",
    )
    fig.update_layout(height=max(400, top_n * 25), yaxis={"categoryorder": "total ascending"},
                      yaxis_title="", xaxis_title=sort_by)
    st.plotly_chart(fig, use_container_width=True)


def page_pitcher_rankings(data: dict):
    st.header("投手ランキング（Marcel法 2026予測）")
    mp = data["marcel_pitchers"]
    if mp.empty:
        st.error("データが読み込めませんでした")
        return

    col1, col2 = st.columns(2)
    top_n = col1.slider("表示人数", 5, 50, 20, key="pitcher_rank_n")
    sort_by = col2.selectbox("ソート", ["ERA", "WHIP", "SO", "W"], key="pitcher_rank_sort")

    ascending = sort_by in ("ERA", "WHIP")
    df = mp[mp["IP"] >= 50].sort_values(sort_by, ascending=ascending).head(top_n).reset_index(drop=True)
    df.index = df.index + 1
    df.index.name = "順位"

    display_cols = ["player", "team", "ERA", "WHIP", "SO", "W", "IP"]
    available = [c for c in display_cols if c in df.columns]
    st.dataframe(df[available], use_container_width=True)

    fig = px.bar(
        df.head(top_n),
        y="player",
        x=sort_by,
        color="team",
        color_discrete_map=NPB_TEAM_COLORS,
        orientation="h",
        template="plotly_white",
    )
    order = "total descending" if ascending else "total ascending"
    fig.update_layout(height=max(400, top_n * 25), yaxis={"categoryorder": order},
                      yaxis_title="", xaxis_title=sort_by)
    st.plotly_chart(fig, use_container_width=True)


def page_pythagorean_standings(data: dict):
    st.header("ピタゴラス順位表")
    pyth = data["pythagorean"]
    if pyth.empty:
        st.error("データが読み込めませんでした")
        return

    years = sorted(pyth["year"].unique())
    year = st.selectbox("年度", [int(y) for y in years], index=len(years) - 1, key="pyth_year")
    df = pyth[pyth["year"] == year].copy()

    for league, label in [("CL", "セ・リーグ"), ("PL", "パ・リーグ")]:
        lg = df[df["league"] == league].sort_values("pyth_WPCT_npb", ascending=False).reset_index(drop=True)
        if lg.empty:
            continue
        lg.index = lg.index + 1
        lg.index.name = "順位"
        st.subheader(label)
        st.dataframe(
            lg[["team", "W", "L", "actual_WPCT", "pyth_WPCT_npb", "pyth_W_npb", "diff_W_npb", "RS", "RA"]].rename(
                columns={
                    "team": "チーム", "W": "勝", "L": "敗", "actual_WPCT": "実勝率",
                    "pyth_WPCT_npb": "ピタ勝率", "pyth_W_npb": "期待勝数",
                    "diff_W_npb": "差", "RS": "得点", "RA": "失点",
                }
            ),
            use_container_width=True,
        )

        fig = go.Figure()
        fig.add_trace(go.Bar(
            name="実際の勝数", x=lg["team"], y=lg["W"],
            marker_color=[NPB_TEAM_COLORS.get(t, "#333") for t in lg["team"]],
        ))
        fig.add_trace(go.Bar(
            name="期待勝数", x=lg["team"], y=lg["pyth_W_npb"],
            marker_color="#9E9E9E",
        ))
        for _, r in lg.iterrows():
            diff = r["diff_W_npb"]
            fig.add_annotation(
                x=r["team"], y=max(r["W"], r["pyth_W_npb"]) + 2,
                text=f"{diff:+.1f}", showarrow=False, font=dict(size=11),
            )
        fig.update_layout(barmode="group", template="plotly_white", height=350,
                          yaxis_title="勝数", legend=dict(orientation="h", y=1.12))
        st.plotly_chart(fig, use_container_width=True)


def page_team_simulation(data: dict):
    st.header("チーム編成シミュレーション")
    pyth = data["pythagorean"]
    saber = data["sabermetrics"]
    marcel_h = data["marcel_hitters"]

    if pyth.empty or saber.empty:
        st.error("データが読み込めませんでした")
        return

    col1, col2 = st.columns(2)
    team = col1.selectbox("チーム", TEAMS, key="sim_team")
    year = col2.slider("ベース年度", 2015, 2025, 2025, key="sim_year")

    mask = pyth["team"].str.contains(_norm(team), na=False) & (pyth["year"] == year)
    team_data = pyth[mask]
    if team_data.empty:
        st.warning(f"{team} ({year}) のデータがありません")
        return

    row = team_data.iloc[0]
    rs = float(row["RS"])
    ra = float(row["RA"])
    games = int(row["G"])

    # チームの選手リスト取得
    team_saber = saber[(saber["team"].str.contains(_norm(team), na=False)) & (saber["year"] == year)]
    team_players = sorted(team_saber["player"].unique().tolist()) if not team_saber.empty else []

    st.markdown("**選手の入れ替え**")
    col_rm, col_add = st.columns(2)
    remove_players = col_rm.multiselect("除外する選手", team_players, key="sim_remove")

    # 追加選手候補: 全選手（チーム外含む）
    all_players = sorted(saber["player"].unique().tolist())
    add_players = col_add.multiselect("追加する選手（他チーム可）", all_players, key="sim_add")

    # wRAA取得ロジック（api.pyと同じ）
    def get_wraa(name: str, team_name: str | None, yr: int | None) -> tuple[str, float, str]:
        matched = _search(saber, name)
        if team_name:
            tm = matched[matched["team"].str.contains(_norm(team_name), na=False)]
            if not tm.empty:
                matched = tm
        if yr is not None:
            ym = matched[matched["year"] == yr]
            if not ym.empty:
                matched = ym
        if not matched.empty:
            r = matched.sort_values("year", ascending=False).iloc[0]
            return r["player"], float(r["wRAA"]), f"{int(r['year'])}実績"
        marcel_match = _search(marcel_h, name)
        if not marcel_match.empty:
            r = marcel_match.iloc[0]
            pa = float(r["PA"])
            ops = float(r["OPS"])
            wraa_est = (ops - 0.700) * pa / 3.2
            return r["player"], round(wraa_est, 1), "Marcel推定"
        return name, 0.0, "データなし"

    # 計算
    rs_adj = rs
    removed_info = []
    for p in remove_players:
        pname, wraa, src = get_wraa(p, team, year)
        removed_info.append({"選手": pname, "wRAA": round(wraa, 1), "ソース": src})
        rs_adj -= wraa

    added_info = []
    for p in add_players:
        pname, wraa, src = get_wraa(p, None, year)
        added_info.append({"選手": pname, "wRAA": round(wraa, 1), "ソース": src})
        rs_adj += wraa

    orig_wpct = _pythagorean_wpct(rs, ra)
    orig_wins = orig_wpct * games
    new_wpct = _pythagorean_wpct(rs_adj, ra)
    new_wins = new_wpct * games
    win_diff = new_wins - orig_wins

    if removed_info:
        st.markdown("**除外選手**")
        st.dataframe(pd.DataFrame(removed_info), use_container_width=True, hide_index=True)
    if added_info:
        st.markdown("**追加選手**")
        st.dataframe(pd.DataFrame(added_info), use_container_width=True, hide_index=True)

    st.markdown("---")
    st.subheader("シミュレーション結果")
    cols = st.columns(3)
    cols[0].metric("ピタゴラス勝率", f"{new_wpct:.3f}", delta=f"{new_wpct - orig_wpct:+.3f}")
    cols[1].metric("期待勝数", f"{new_wins:.1f}", delta=f"{win_diff:+.1f}")
    cols[2].metric("調整後得点", f"{rs_adj:.0f}", delta=f"{rs_adj - rs:+.0f}")

    fig = go.Figure(data=[
        go.Bar(name="現状", x=["期待勝数"], y=[orig_wins],
               marker_color=NPB_TEAM_COLORS.get(team, "#333")),
        go.Bar(name="シミュレーション", x=["期待勝数"], y=[new_wins],
               marker_color="#FF9800"),
    ])
    fig.update_layout(barmode="group", template="plotly_white", height=300, yaxis_title="勝数")
    st.plotly_chart(fig, use_container_width=True)


# --- メイン ---


def main():
    st.set_page_config(page_title="NPB成績予測", page_icon="&#9918;", layout="wide")
    st.title("NPB成績予測ダッシュボード")
    st.caption(
        "データソース: [プロ野球データFreak](https://baseball-data.com) / "
        "[日本野球機構 NPB](https://npb.jp)"
    )

    data = load_all()

    page = st.sidebar.radio(
        "ページ選択",
        [
            "打者予測",
            "投手予測",
            "チーム勝率",
            "セイバーメトリクス",
            "打者ランキング",
            "投手ランキング",
            "ピタゴラス順位表",
            "チームシミュレーション",
        ],
    )

    pages = {
        "打者予測": page_hitter_prediction,
        "投手予測": page_pitcher_prediction,
        "チーム勝率": page_team_wpct,
        "セイバーメトリクス": page_sabermetrics,
        "打者ランキング": page_hitter_rankings,
        "投手ランキング": page_pitcher_rankings,
        "ピタゴラス順位表": page_pythagorean_standings,
        "チームシミュレーション": page_team_simulation,
    }

    pages[page](data)


if __name__ == "__main__":
    main()
